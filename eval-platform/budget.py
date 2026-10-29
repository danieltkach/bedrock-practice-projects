"""Hard spending cap. Checked BEFORE every paid call, so a run aborts instead of overspending."""
import threading


class BudgetExceeded(Exception):
    """Raised when the next call could push the estimated spend above the cap."""


class Budget:
    def __init__(self, cap_usd):
        self.cap = cap_usd
        self.spent = 0.0
        self._lock = threading.Lock()   # many threads share one budget

    def check(self, worst_case_usd):
        """Raise if spending `worst_case_usd` more could cross the cap."""
        with self._lock:
            if self.spent + worst_case_usd > self.cap:
                raise BudgetExceeded(
                    f"budget cap ${self.cap:.2f} would be exceeded (spent ${self.spent:.4f}, "
                    f"next call up to ${worst_case_usd:.4f})")

    def add(self, usd):
        with self._lock:
            self.spent += usd
