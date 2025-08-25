from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Optional

# =========================
# Configurazione di gioco
# =========================
BOARD_LEN = 16  # colonne 1..16
LEFT_BASE_X = 1
RIGHT_BASE_X = 16
START_A_X = 2
START_B_X = 15

RIGHT = +1
LEFT  = -1


class Action(Enum):
    CARICA    = "carica"
    ATTACCA   = "attacca"
    MOVIMENTO = "movimento"
    DIFENDI   = "difendi"
    RITIRATA  = "ritirata"


ACTION_MODS: Dict[Action, Dict[str, float]] = {
    Action.CARICA:    {"vel": 1.00, "atk": 1.25, "def": 0.00},
    Action.ATTACCA:   {"vel": 0.25, "atk": 1.00, "def": 0.25},
    Action.MOVIMENTO: {"vel": 0.50, "atk": 0.25, "def": 0.25},
    Action.DIFENDI:   {"vel": 0.00, "atk": 0.25, "def": 1.00},
    Action.RITIRATA:  {"vel": 0.75, "atk": 0.00, "def": 0.00},
}


def clamp_board(x: int) -> int:
    return max(1, min(BOARD_LEN, x))


@dataclass
class Unit:
    id: str
    faction: str              # "A" o "B"
    salute: int               # HP
    attacco: int
    difesa: int
    direzione: int            # +1 destra / -1 sinistra (direzione naturale)
    pos: int                  # POSIZIONE INTERA: 1..16
    last_pos: int
    next_action: Action = Action.MOVIMENTO
    prev_action: Optional[Action] = None

    @property
    def at_base_enemy(self) -> bool:
        if self.faction == "A":
            return self.pos == RIGHT_BASE_X
        else:
            return self.pos == LEFT_BASE_X

    def speed_value(self) -> float:
        return ACTION_MODS[self.next_action]["vel"]

    def step_size(self) -> int:
        return 0 if self.next_action == Action.DIFENDI else 1

    def intended_direction(self) -> int:
        return -self.direzione if self.next_action == Action.RITIRATA else self.direzione

    def att_mod(self) -> float:
        return ACTION_MODS[self.next_action]["atk"]

    def def_mod(self) -> float:
        return ACTION_MODS[self.next_action]["def"]


@dataclass
class Board:
    length: int = BOARD_LEN

    def occupants(self, units: List[Unit], x: int) -> List[Unit]:
        return [u for u in units if u.salute > 0 and u.pos == x]


@dataclass
class Game:
    board: Board
    units: List[Unit] = field(default_factory=list)
    turn_count: int = 0
    battle_log: List[str] = field(default_factory=list)

    # ================ Utility =================

    def get_units_alive(self) -> List[Unit]:
        return [u for u in self.units if u.salute > 0]

    def factions(self) -> Dict[str, List[Unit]]:
        alive = self.get_units_alive()
        return {"A": [u for u in alive if u.faction == "A"],
                "B": [u for u in alive if u.faction == "B"]}

    def set_actions(self, actions_by_id: Dict[str, Action]) -> None:
        for u in self.get_units_alive():
            if u.id in actions_by_id:
                u.prev_action = u.next_action
                u.next_action = actions_by_id[u.id]

    # Helper: niente battaglia se coppia ritirata vs attacco
    @staticmethod
    def _is_retreat_vs_attack(u: Unit, v: Unit) -> bool:
        a, b = u.next_action, v.next_action
        return (a == Action.RITIRATA and b == Action.ATTACCA) or (b == Action.RITIRATA and a == Action.ATTACCA)

    # ================ Turno ====================

    def step(self) -> None:
        self.turn_count += 1
        self.battle_log.clear()

        alive = self.get_units_alive()
        if not alive:
            return

        intents = self._compute_intents(alive)
        battles, winners, blocked = self._resolve_blocking_and_contests(alive, intents)
        self._resolve_battles(battles)
        self._apply_movements(alive, intents, winners, blocked)

    # ============== Intenzioni =================

    def _compute_intents(self, alive: List[Unit]) -> Dict[str, int]:
        intents: Dict[str, int] = {}
        for u in alive:
            step = u.step_size()
            if step == 0:
                intents[u.id] = u.pos
                continue
            d = u.intended_direction()
            intents[u.id] = clamp_board(u.pos + d * step)
        return intents

    # ====== Blocco su celle occupate & contese ========

    def _resolve_blocking_and_contests(
        self,
        alive: List[Unit],
        intents: Dict[str, int]
    ) -> Tuple[List[Tuple[Unit, Unit]], Dict[int, Unit], set[str]]:
        """
        Restituisce:
          - battles: lista di coppie (u,v) che combatteranno (danni simultanei).
          - winners: mappa cella_dest -> unità vincitrice (solo per celle libere a inizio turno).
          - blocked: insieme di unit_id bloccati (non si muovono in questo turno).

        Regole centrali:
          * Una unità NON può mai entrare in una cella che era OCCUPATA a inizio turno (amico o nemico).
            - Se il tentativo è su cella occupata da nemico: battaglia & bloccato (ECCEZIONE: Attacca vs Ritirata -> no battaglia).
            - Se occupata da amico: solo bloccato.
          * Non si può sorpassare in alcun modo (niente swap/incrocio).
          * Se 2+ unità vogliono la STESSA cella LIBERA:
              - vince la più veloce -> occuperà la cella;
              - in pareggio: nessuno si muove; se fazioni opposte -> battaglia (ECCEZIONE: Attacca vs Ritirata -> no battaglia).
          * Stessa direzione: se l'inseguitore è più veloce del davanti -> battaglia & fermi (senza superare).
        """
        id2u = {u.id: u for u in alive}
        battles: List[Tuple[Unit, Unit]] = []
        winners: Dict[int, Unit] = {}
        blocked: set[str] = set()

        # Occupazione iniziale (fotografia a inizio turno)
        initial_occ: Dict[int, Unit] = {}
        for u in alive:
            initial_occ[u.pos] = u  # 1D: max 1 per cella

        # 1) Gruppi per destinazione
        dest_groups: Dict[int, List[Unit]] = {}
        for u in alive:
            dest_groups.setdefault(intents[u.id], []).append(u)

        # 2) Contese
        for cell, group in dest_groups.items():
            # Se la cella era occupata all'inizio: NESSUNO può entrarci questo turno
            if cell in initial_occ:
                occupant = initial_occ[cell]
                for g in group:
                    if g.id == occupant.id:
                        continue  # restare nella stessa cella (difendi)
                    blocked.add(g.id)
                    if g.faction != occupant.faction and not self._is_retreat_vs_attack(g, occupant):
                        battles.append((g, occupant))
                continue

            # Cella inizialmente libera
            if len(group) >= 2:
                max_vel = max(g.speed_value() for g in group)
                contenders = [g for g in group if abs(g.speed_value() - max_vel) < 1e-9]
                if len(contenders) == 1:
                    winner = contenders[0]
                    winners[cell] = winner
                    # Gli altri non si muovono; battaglia solo se non è (ritirata vs attacca)
                    for g in group:
                        if g.id != winner.id:
                            blocked.add(g.id)
                            if g.faction != winner.faction and not self._is_retreat_vs_attack(winner, g):
                                battles.append((winner, g))
                else:
                    # Pareggio: nessuno si muove; battaglia tra opposti tranne coppie (ritirata vs attacca)
                    for g in group:
                        blocked.add(g.id)
                    A_side = [g for g in contenders if g.faction == "A"]
                    B_side = [g for g in contenders if g.faction == "B"]
                    for a in A_side:
                        for b in B_side:
                            if not self._is_retreat_vs_attack(a, b):
                                battles.append((a, b))

        # 3) Tentativi singoli su cella occupata a inizio turno (non già marcati sopra)
        for u in alive:
            if u.id in blocked:
                continue
            dest = intents[u.id]
            # se la cella era occupata a inizio turno da altri -> bloccato
            if dest in initial_occ and initial_occ[dest].id != u.id:
                blocked.add(u.id)
                if initial_occ[dest].faction != u.faction and not self._is_retreat_vs_attack(u, initial_occ[dest]):
                    battles.append((u, initial_occ[dest]))
                continue

            # se la cella ha già un winner diverso -> bloccato; se avversario -> battaglia (salvo ritirata vs attacca)
            if dest in winners and winners[dest].id != u.id:
                blocked.add(u.id)
                if winners[dest].faction != u.faction and not self._is_retreat_vs_attack(u, winners[dest]):
                    battles.append((u, winners[dest]))
                continue

        # 4) Inseguimento stessa direzione (senza superare mai) — solo tra amici
        for u in alive:
            if u.id in blocked:
                continue
            d = u.intended_direction()
            front_cell = u.pos + d  # cella immediatamente davanti a inizio turno
            front = initial_occ.get(front_cell)
            if front and front.faction == u.faction:
                if id2u[front.id].intended_direction() == d:
                    if u.speed_value() > id2u[front.id].speed_value():
                        blocked.add(u.id)
                        blocked.add(front.id)
                        # stessi colori: nessuna battaglia (regola generale del tuo sistema)
                        # se vuoi far scattare "ammaccature" anche tra compagni, qui potresti aggiungere una battaglia.

        # 5) Incrocio (swap) resta implicitamente bloccato dalla regola "cella occupata a inizio turno"

        return battles, winners, blocked

    # ============== Battaglie =================

    def _resolve_battles(self, battles: List[Tuple[Unit, Unit]]) -> None:
        # danni simultanei, deduplicati
        seen = set()
        for u, v in battles:
            if u.salute <= 0 or v.salute <= 0:
                continue
            key = tuple(sorted((u.id, v.id)))
            if key in seen:
                continue
            seen.add(key)
            u_atk = max(0.0, u.attacco * u.att_mod() - v.difesa * v.def_mod())
            v_atk = max(0.0, v.attacco * v.att_mod() - u.difesa * u.def_mod())
            u_dmg = int(round(v_atk))
            v_dmg = int(round(u_atk))
            if u_dmg > 0 or v_dmg > 0:
                self.battle_log.append(
                    f"{u.id} ({u.faction}) vs {v.id} ({v.faction}) -> "
                    f"danni: {u.id} -{u_dmg}, {v.id} -{v_dmg}"
                )
            u.salute = max(0, u.salute - u_dmg)
            v.salute = max(0, v.salute - v_dmg)

    # ============== Applicazione move =========

    def _apply_movements(
        self,
        alive: List[Unit],
        intents: Dict[str, int],
        winners: Dict[int, Unit],
        blocked: set[str]
    ) -> None:
        # Applica gli spostamenti:
        # - solo i winners (celle libere a inizio turno con contesa vinta) si muovono nella cella
        # - gli altri non bloccati si muovono verso la loro destinazione SOLO se era libera a inizio turno e non contesa
        initial_free = {x for x in range(1, BOARD_LEN + 1)}
        for u in alive:
            if u.pos in initial_free:
                initial_free.remove(u.pos)

        for u in alive:
            u.last_pos = u.pos

        # 1) winners
        for cell, w in winners.items():
            w.pos = cell

        # 2) mosse semplici (no blocchi, no winners, destinazione era libera a inizio turno)
        winner_ids = {w.id for w in winners.values()}
        for u in alive:
            if u.id in blocked or u.id in winner_ids:
                continue
            dest = intents[u.id]
            if dest not in initial_free and dest != u.pos:
                continue  # la cella non era libera a inizio turno
            if any(w.pos == dest for w in winners.values()):
                continue
            u.pos = dest

    # ============== Stato & UI =================

    def check_victory(self) -> Optional[str]:
        for u in self.get_units_alive():
            if u.at_base_enemy:
                return u.faction
        return None

    def ascii_board(self) -> str:
        cells: List[str] = []
        alive = self.get_units_alive()
        for x in range(1, self.board.length + 1):
            occ = self.board.occupants(alive, x)
            if not occ:
                cells.append("[  ]")
            else:
                u = occ[0]
                cells.append(f"[{u.faction}{u.id[-1]}]")
        return "".join(cells)

    def print_status(self) -> None:
        print(f"\n--- TURNO {self.turn_count} ---")
        print(self.ascii_board())
        for u in self.units:
            side = "→" if u.direzione == RIGHT else "←"
            print(f" {u.id}({u.faction}{side})  HP:{u.salute:>3}  pos:{u.pos:>2d}  azione:{u.next_action.value}")
        if self.battle_log:
            print(" Battaglie:")
            for line in self.battle_log:
                print("  -", line)


# =========================
# Esempio di utilizzo CLI
# =========================

def make_default_game() -> Game:
    game = Game(Board())
    game.units.append(Unit(
        id="A1", faction="A", salute=160, attacco=48, difesa=32,
        direzione=RIGHT, pos=START_A_X, last_pos=START_A_X
    ))
    game.units.append(Unit(
        id="B1", faction="B", salute=160, attacco=48, difesa=32,
        direzione=LEFT, pos=START_B_X, last_pos=START_B_X
    ))
    return game

ACTION_PARSE = {
    "c": Action.CARICA,
    "a": Action.ATTACCA,
    "m": Action.MOVIMENTO,
    "d": Action.DIFENDI,
    "r": Action.RITIRATA,
}

def prompt_actions(game: Game) -> Dict[str, Action]:
    actions: Dict[str, Action] = {}
    alive = [u for u in game.units if u.salute > 0]
    print("\nScegli azioni (c=carica, a=attacca, m=movimento, d=difendi, r=ritirata).")
    for u in alive:
        while True:
            choice = input(f"  Azione per {u.id} ({u.faction}): ").strip().lower()
            if choice in ACTION_PARSE:
                actions[u.id] = ACTION_PARSE[choice]
                break
            print("  Input non valido.")
    return actions

def main():
    game = make_default_game()
    print("Gioco 'Linea 16x1 — blocco celle occupate, no sorpasso, ritirata > attacco (no battaglia)'.")
    print("Vince chi occupa la base nemica (A -> x=16, B -> x=1).")
    print("Legenda: [  ] vuota, [A1]/[B1] occupata.\n")

    game.print_status()

    while True:
        actions = prompt_actions(game)
        game.set_actions(actions)
        game.step()
        game.print_status()

        winner = game.check_victory()
        if winner:
            print(f"\n>>> Vittoria della fazione {winner}! <<<")
            break

        factions = game.factions()
        if not factions["A"]:
            print("\n>>> Fazione B vince (A eliminata)! <<<")
            break
        if not factions["B"]:
            print("\n>>> Fazione A vince (B eliminata)! <<<")
            break

if __name__ == "__main__":
    main()
