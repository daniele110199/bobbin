"""Serve user-uploaded files."""
import os
import subprocess

UPLOAD_DIR = "/var/app/uploads"


def read_upload(filename):
    path = os.path.join(UPLOAD_DIR, filename)
    with open(path) as fh:
        return fh.read()


def convert_to_pdf(filename):
    cmd = "libreoffice --headless --convert-to pdf " + filename
    subprocess.call(cmd, shell=True)
