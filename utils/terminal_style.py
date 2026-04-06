from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.style import Style
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
import time

class TerminalStyle:
    def __init__(self):
        self.console = Console()
        self.primary_color = "cyan"
        self.secondary_color = "magenta"
        self.accent_color = "bold green"

    def print_header(self, task_title):
        """Prints a professional branded header for the task."""
        banner = Text(justify="center")
        banner.append("🤖 AI INTERNSHIP @ CODETECH SOLUTIONS\n", style="bold cyan")
        banner.append(f"TASK: {task_title.upper()}", style="bold white")
        
        self.console.print(Panel(banner, border_style="blue", expand=False, padding=(1, 2)))

    def print_input_panel(self, content, label="INPUT DATA"):
        """Prints input data in a clean panel."""
        self.console.print(Panel(
            content,
            title=f"[bold {self.primary_color}]{label}[/]",
            border_style=self.primary_color,
            padding=(1, 2)
        ))

    def print_output_panel(self, content, label="MODEL OUTPUT"):
        """Prints model output in a visually distinct panel."""
        self.console.print(Panel(
            content,
            title=f"[bold {self.accent_color}]{label}[/]",
            border_style=self.accent_color,
            padding=(1, 2)
        ))

    def print_metrics(self, metrics_title, metrics_dict):
        """Prints a table of metrics or statistics."""
        table = Table(title=metrics_title, show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="dim", width=20)
        table.add_column("Value", justify="right")

        for key, value in metrics_dict.items():
            table.add_row(str(key), str(value))

        self.console.print(table)

    def print_info(self, message):
        """Prints an informational message with a nice icon."""
        self.console.print(f"[bold blue]ℹ[/] [italic gray]{message}[/]")

    def print_success(self, message):
        """Prints a success message."""
        self.console.print(f"[bold green]✔[/] {message}")

    def get_progress(self, description="Processing..."):
        """Returns a stylized progress bar context manager."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            console=self.console
        )

# Global instance for easy use
style = TerminalStyle()
