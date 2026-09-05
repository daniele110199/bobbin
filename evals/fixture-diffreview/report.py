"""Generate a report from a template."""
import subprocess


def render(template_path):
    subprocess.run(["pandoc", template_path, "-o", "out.pdf"], check=True)


def archive(name):
    subprocess.run("tar czf backup.tgz " + name, shell=True)
