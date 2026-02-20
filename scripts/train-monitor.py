#!/usr/bin/env python3
"""Monitor wandb training runs and send Telegram notifications."""

import argparse
import os
import time

import wandb


def load_env():
    """Load credentials from .env file."""
    from pathlib import Path
    from dotenv import load_dotenv, find_dotenv
    # Try scripts/.env first (where tpu-watcher keeps it), then search upwards
    scripts_env = Path(__file__).parent / ".env"
    if scripts_env.exists():
        load_dotenv(scripts_env)
    else:
        load_dotenv(find_dotenv(usecwd=True))


def escape_markdown(text: str) -> str:
    """Escape Telegram Markdown special characters."""
    return text.replace('_', '\\_')


def send_telegram(message: str) -> bool:
    """Send message via Telegram bot."""
    import urllib.request
    import urllib.parse

    bot_token = os.environ.get("BOT_TOKEN")
    chat_id = os.environ.get("CHAT_ID")

    if not bot_token or not chat_id:
        print("Warning: BOT_TOKEN or CHAT_ID not set, skipping Telegram")
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }).encode()

    try:
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        urllib.request.urlopen(url, data, timeout=10, context=ctx)
        return True
    except Exception as e:
        print(f"Telegram error: {e}")
        return False


def send_telegram_photo(image_path: str, caption: str = "") -> bool:
    """Send photo via Telegram bot."""
    import urllib.request
    import mimetypes
    import uuid

    bot_token = os.environ.get("BOT_TOKEN")
    chat_id = os.environ.get("CHAT_ID")

    if not bot_token or not chat_id:
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"

    # Build multipart form data
    boundary = uuid.uuid4().hex
    lines = []

    # Add chat_id field
    lines.append(f"--{boundary}")
    lines.append('Content-Disposition: form-data; name="chat_id"')
    lines.append("")
    lines.append(chat_id)

    # Add caption field
    if caption:
        lines.append(f"--{boundary}")
        lines.append('Content-Disposition: form-data; name="caption"')
        lines.append("")
        lines.append(caption)

        # Add parse_mode for caption
        lines.append(f"--{boundary}")
        lines.append('Content-Disposition: form-data; name="parse_mode"')
        lines.append("")
        lines.append("Markdown")

    # Add photo field
    lines.append(f"--{boundary}")
    lines.append(f'Content-Disposition: form-data; name="photo"; filename="plot.png"')
    lines.append("Content-Type: image/png")
    lines.append("")

    header = "\r\n".join(lines).encode() + b"\r\n"
    footer = f"\r\n--{boundary}--\r\n".encode()

    with open(image_path, "rb") as f:
        image_data = f.read()

    body = header + image_data + footer

    try:
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
        )
        urllib.request.urlopen(req, timeout=30, context=ctx)
        return True
    except Exception as e:
        print(f"Telegram photo error: {e}")
        return False


def ema(data, alpha: float = 0.1):
    """Exponential moving average."""
    import numpy as np
    data = np.array(data)
    result = np.zeros_like(data)
    result[0] = data[0]
    for i in range(1, len(data)):
        if np.isnan(data[i]):
            result[i] = result[i-1]
        else:
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
    return result


def create_training_plot(run, num_samples: int = 500) -> str | None:
    """Create a combined plot of loss, lr, grad_norm and save to temp file."""
    import tempfile
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return None

    # Fetch history (500 samples for smooth EMA)
    history = run.history(samples=num_samples, keys=["train/loss", "train/learning_rate", "train/grad_norm", "_step"])
    if history.empty:
        return None

    fig, ax1 = plt.subplots(figsize=(10, 5))
    fig.suptitle(f"{run.name}", fontsize=12)

    steps = history["_step"].values

    # Loss (left y-axis, log scale, blue)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss", color="tab:blue")
    loss_raw_last, loss_ema_last = None, None
    if "train/loss" in history:
        loss = history["train/loss"].values
        loss_smooth = ema(loss, alpha=0.05)
        loss_raw_last = loss[~np.isnan(loss)][-1] if len(loss[~np.isnan(loss)]) > 0 else None
        loss_ema_last = loss_smooth[-1]
        ax1.semilogy(steps, loss_smooth, "tab:blue", linewidth=1.5, label="Loss (EMA)")
        ax1.semilogy(steps, loss, "tab:blue", linewidth=0.3, alpha=0.3)
        # Better log scale ticks - label at every grid line
        from matplotlib.ticker import LogLocator, FuncFormatter
        ax1.yaxis.set_major_locator(LogLocator(base=10, subs=np.arange(1, 10), numticks=20))
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2g}'))
        min_loss = np.nanmin(loss)
        ax1.set_ylim(bottom=min_loss / 1.1)  # Dynamic: min loss / 1.1
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3, which='both')

    # Learning rate (right y-axis 1, green)
    ax2 = ax1.twinx()
    ax2.set_ylabel("LR", color="tab:green")
    if "train/learning_rate" in history:
        lr = history["train/learning_rate"].values
        ax2.plot(steps, lr, "tab:green", linewidth=1.5, label="LR")
    ax2.tick_params(axis="y", labelcolor="tab:green")
    ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

    # Grad norm (right y-axis 2, red, offset)
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 60))
    ax3.set_ylabel("Grad Norm", color="tab:red")
    grad_raw_last, grad_ema_last = None, None
    if "train/grad_norm" in history:
        grad = history["train/grad_norm"].values
        grad_smooth = ema(grad, alpha=0.05)
        grad_raw_last = grad[~np.isnan(grad)][-1] if len(grad[~np.isnan(grad)]) > 0 else None
        grad_ema_last = grad_smooth[-1]
        ax3.plot(steps, grad_smooth, "tab:red", linewidth=1.5, label="Grad (EMA)")
        ax3.plot(steps, grad, "tab:red", linewidth=0.3, alpha=0.3)
    ax3.tick_params(axis="y", labelcolor="tab:red")

    # Add current values annotation inside plot, near latest point
    current_step = int(steps.max()) if len(steps) > 0 else 0
    last_step = steps[-1] if len(steps) > 0 else 0

    # Position text near the last data point
    if loss_ema_last is not None:
        ax1.annotate(
            f"Step {current_step:,}\nLoss: {loss_raw_last:.3f}\n(EMA: {loss_ema_last:.3f})",
            xy=(last_step, loss_ema_last),
            xytext=(-80, 20), textcoords='offset points',
            fontsize=8, color='tab:blue',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='tab:blue'),
            arrowprops=dict(arrowstyle='->', color='tab:blue', alpha=0.5)
        )

    if grad_ema_last is not None:
        ax3.annotate(
            f"Grad: {grad_raw_last:.2f}\n(EMA: {grad_ema_last:.2f})",
            xy=(last_step, grad_ema_last),
            xytext=(-80, -30), textcoords='offset points',
            fontsize=8, color='tab:red',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='tab:red'),
            arrowprops=dict(arrowstyle='->', color='tab:red', alpha=0.5)
        )

    plt.tight_layout()

    # Save to temp file
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(tmp.name, dpi=100, bbox_inches='tight')
    plt.close(fig)

    return tmp.name


def get_active_runs(entity: str, project: str) -> list:
    """Get currently running wandb runs."""
    api = wandb.Api()
    runs = api.runs(
        f"{entity}/{project}",
        filters={"state": "running"},
        order="-created_at"
    )
    return list(runs)


def get_latest_run(entity: str, project: str) -> wandb.apis.public.Run | None:
    """Get the most recent run (running or finished)."""
    api = wandb.Api()
    runs = api.runs(
        f"{entity}/{project}",
        order="-created_at",
        per_page=1
    )
    runs_list = list(runs)
    return runs_list[0] if runs_list else None


def format_metrics(run, last_step: int | None = None) -> str:
    """Format run metrics for Telegram message."""
    summary = run.summary
    if not summary:
        return None

    step = int(summary.get("_step", 0))

    # Skip if no new steps
    if last_step is not None and step <= last_step:
        return None

    safe_name = escape_markdown(run.name)
    lines = [f"*{safe_name}* (step {step})"]

    # Common metrics to report
    metric_names = [
        ("train/loss", "Loss"),
        ("train/learning_rate", "LR"),
        ("train/grad_norm", "Grad norm"),
        ("eval/loss", "Eval loss"),
        ("eval/accuracy", "Accuracy"),
        ("samples_per_second", "Samples/s"),
    ]

    for key, label in metric_names:
        val = summary.get(key)
        if val is not None:
            if isinstance(val, float):
                if val < 0.001:
                    lines.append(f"  {label}: {val:.2e}")
                else:
                    lines.append(f"  {label}: {val:.4f}")
            else:
                lines.append(f"  {label}: {val}")

    return "\n".join(lines) if len(lines) > 1 else None


def monitor_loop(entity: str, project: str, interval: int, notify_every: int):
    """Main monitoring loop."""
    print(f"Monitoring {entity}/{project}")
    print(f"Checking every {interval}s, notifying every {notify_every}s")
    print("Press Ctrl+C to stop\n")

    last_run_id = None
    last_step = None
    last_notified_time = None

    while True:
        try:
            # Check for active runs first, fall back to latest
            runs = get_active_runs(entity, project)
            if runs:
                run = runs[0]
                status = "🟢 running"
            else:
                run = get_latest_run(entity, project)
                if not run:
                    print(f"{time.strftime('%H:%M:%S')}: No runs found")
                    time.sleep(interval)
                    continue
                status = f"⚪ {run.state}"

            # New run started?
            if run.id != last_run_id:
                last_run_id = run.id
                last_step = None
                last_notified_step = None
                safe_name = escape_markdown(run.name)
                safe_project = escape_markdown(project)
                msg = f"📊 Monitoring: *{safe_name}*\n{status}\nProject: {safe_project}"
                send_telegram(msg)
                print(f"{time.strftime('%H:%M:%S')}: Now monitoring {run.name}")

            # Get latest metrics from summary (most recent logged values)
            run.update()  # Refresh to get latest summary
            summary = run.summary
            step = int(summary.get("_step", 0))

            # Log to console
            loss = summary.get("train/loss", "?")
            if isinstance(loss, float):
                loss = f"{loss:.4f}"
            print(f"{time.strftime('%H:%M:%S')}: step {step}, loss {loss}")

            # Notify on time interval
            now = time.time()
            should_notify = (
                last_notified_time is None or
                now - last_notified_time >= notify_every
            )

            if should_notify and step > (last_step or 0):
                msg = format_metrics(run, last_step)
                if msg:
                    # Send plot with caption
                    plot_path = create_training_plot(run)
                    if plot_path:
                        send_telegram_photo(plot_path, caption=msg)
                        os.unlink(plot_path)  # Clean up temp file
                    else:
                        send_telegram(msg)
                    last_notified_time = now

            last_step = step

            # Check if run finished (already refreshed above)
            if run.state in ("finished", "failed", "crashed"):
                emoji = "✅" if run.state == "finished" else "❌"
                safe_name = escape_markdown(run.name)
                msg = f"{emoji} Run *{safe_name}* {run.state}"
                if run.summary:
                    final_loss = run.summary.get("train/loss")
                    if final_loss:
                        msg += f"\nFinal loss: {final_loss:.4f}"
                send_telegram(msg)
                print(f"Run {run.state}, waiting for new run...")
                last_run_id = None

        except KeyboardInterrupt:
            print("\nStopping monitor")
            break
        except Exception as e:
            print(f"Error: {e}")

        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Monitor wandb runs via Telegram")
    parser.add_argument("--entity", default="yepster", help="wandb entity/username")
    parser.add_argument("--project", default="mbert-jax", help="wandb project name")
    parser.add_argument("--interval", type=int, default=60, help="Check interval in seconds")
    parser.add_argument("--notify-every", type=int, default=600, help="Notify every N seconds")
    parser.add_argument("--test", action="store_true", help="Send test message and exit")
    args = parser.parse_args()

    load_env()

    if args.test:
        send_telegram("🧪 Train monitor test message")
        print("Test message sent!")
        return

    monitor_loop(args.entity, args.project, args.interval, args.notify_every)


if __name__ == "__main__":
    main()