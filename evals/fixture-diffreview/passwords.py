"""Password storage."""
import bcrypt


def store(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())


def verify(password, stored):
    return bcrypt.checkpw(password.encode(), stored)
