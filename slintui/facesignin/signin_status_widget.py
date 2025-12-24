import datetime
import slint


logs_model: slint.ListModel[str] = slint.ListModel([])


def init_signin_status(window) -> None:
    window.signin_status_text = "等待签到"
    window.signin_person_text = ""
    window.signin_logs = logs_model


def append_log(name: str, score: float) -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    logs_model.append(f"{ts} {name} {score:.2f}")
