"""Unit tests for the authentication module."""
import os
import tempfile
import unittest

from src.auth import (
    User, Session, register_user, login_user, get_session,
    get_user, logout_user, logout_all_sessions, cleanup_expired_sessions,
    change_password, update_username, delete_user, init_auth_db
)
from src.errors import AuthError


class TestAuth(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.tmp.name, "auth.db")

    def tearDown(self):
        self.tmp.cleanup()

    def test_register_user(self):
        user = register_user(self.db, "test@example.com", "testuser", "password123")
        self.assertIsInstance(user, User)
        self.assertEqual(user.email, "test@example.com")
        self.assertEqual(user.username, "testuser")
        self.assertIsNotNone(user.created_at)

    def test_register_duplicate_email(self):
        register_user(self.db, "test@example.com", "user1", "password123")
        with self.assertRaises(AuthError) as ctx:
            register_user(self.db, "test@example.com", "user2", "password123")
        self.assertIn("Email already registered", str(ctx.exception))

    def test_register_duplicate_username(self):
        register_user(self.db, "test1@example.com", "testuser", "password123")
        with self.assertRaises(AuthError) as ctx:
            register_user(self.db, "test2@example.com", "testuser", "password123")
        self.assertIn("Username already taken", str(ctx.exception))

    def test_register_invalid_email(self):
        with self.assertRaises(AuthError) as ctx:
            register_user(self.db, "invalid-email", "testuser", "password123")
        self.assertIn("Invalid email", str(ctx.exception))

    def test_register_short_username(self):
        with self.assertRaises(AuthError) as ctx:
            register_user(self.db, "test@example.com", "ab", "password123")
        self.assertIn("at least 3 characters", str(ctx.exception))

    def test_register_short_password(self):
        with self.assertRaises(AuthError) as ctx:
            register_user(self.db, "test@example.com", "testuser", "short")
        self.assertIn("at least 8 characters", str(ctx.exception))

    def test_login_user(self):
        register_user(self.db, "test@example.com", "testuser", "password123")
        user, session = login_user(self.db, "test@example.com", "password123")
        self.assertIsInstance(user, User)
        self.assertIsInstance(session, Session)
        self.assertEqual(user.email, "test@example.com")
        self.assertEqual(session.user_id, user.id)
        self.assertIsNotNone(session.token)

    def test_login_wrong_password(self):
        register_user(self.db, "test@example.com", "testuser", "password123")
        with self.assertRaises(AuthError) as ctx:
            login_user(self.db, "test@example.com", "wrongpassword")
        self.assertIn("Invalid email or password", str(ctx.exception))

    def test_login_nonexistent_user(self):
        with self.assertRaises(AuthError) as ctx:
            login_user(self.db, "nonexistent@example.com", "password123")
        self.assertIn("Invalid email or password", str(ctx.exception))

    def test_get_session_valid(self):
        register_user(self.db, "test@example.com", "testuser", "password123")
        user, session = login_user(self.db, "test@example.com", "password123")
        retrieved = get_session(self.db, session.token)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.token, session.token)
        self.assertEqual(retrieved.user_id, user.id)

    def test_get_session_invalid_token(self):
        result = get_session(self.db, "invalid-token")
        self.assertIsNone(result)

    def test_get_user(self):
        register_user(self.db, "test@example.com", "testuser", "password123")
        user, _ = login_user(self.db, "test@example.com", "password123")
        retrieved = get_user(self.db, user.id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.id, user.id)
        self.assertEqual(retrieved.email, user.email)

    def test_logout_user(self):
        register_user(self.db, "test@example.com", "testuser", "password123")
        _, session = login_user(self.db, "test@example.com", "password123")
        result = logout_user(self.db, session.token)
        self.assertTrue(result)
        # Session should be gone
        self.assertIsNone(get_session(self.db, session.token))

    def test_logout_all_sessions(self):
        register_user(self.db, "test@example.com", "testuser", "password123")
        user, _ = login_user(self.db, "test@example.com", "password123")
        # Create multiple sessions
        login_user(self.db, "test@example.com", "password123")
        login_user(self.db, "test@example.com", "password123")
        count = logout_all_sessions(self.db, user.id)
        self.assertEqual(count, 3)
        # All sessions should be gone
        self.assertIsNone(get_session(self.db, "any-token"))

    def test_change_password(self):
        register_user(self.db, "test@example.com", "testuser", "password123")
        user, _ = login_user(self.db, "test@example.com", "password123")
        change_password(self.db, user.id, "password123", "newpassword123")
        # Old password should not work
        with self.assertRaises(AuthError):
            login_user(self.db, "test@example.com", "password123")
        # New password should work
        user2, session2 = login_user(self.db, "test@example.com", "newpassword123")
        self.assertEqual(user2.id, user.id)

    def test_change_password_wrong_old(self):
        register_user(self.db, "test@example.com", "testuser", "password123")
        user, _ = login_user(self.db, "test@example.com", "password123")
        with self.assertRaises(AuthError) as ctx:
            change_password(self.db, user.id, "wrongold", "newpassword123")
        self.assertIn("Current password is incorrect", str(ctx.exception))

    def test_change_password_short_new(self):
        register_user(self.db, "test@example.com", "testuser", "password123")
        user, _ = login_user(self.db, "test@example.com", "password123")
        with self.assertRaises(AuthError) as ctx:
            change_password(self.db, user.id, "password123", "short")
        self.assertIn("at least 8 characters", str(ctx.exception))

    def test_update_username(self):
        register_user(self.db, "test@example.com", "testuser", "password123")
        user, _ = login_user(self.db, "test@example.com", "password123")
        update_username(self.db, user.id, "newusername")
        retrieved = get_user(self.db, user.id)
        self.assertEqual(retrieved.username, "newusername")

    def test_update_username_taken(self):
        register_user(self.db, "test1@example.com", "user1", "password123")
        user2, _ = login_user(self.db, "test1@example.com", "password123")
        register_user(self.db, "test2@example.com", "user2", "password123")
        with self.assertRaises(AuthError) as ctx:
            update_username(self.db, user2.id, "user1")
        self.assertIn("Username already taken", str(ctx.exception))

    def test_delete_user(self):
        register_user(self.db, "test@example.com", "testuser", "password123")
        user, _ = login_user(self.db, "test@example.com", "password123")
        delete_user(self.db, user.id, "password123")
        self.assertIsNone(get_user(self.db, user.id))

    def test_delete_user_wrong_password(self):
        register_user(self.db, "test@example.com", "testuser", "password123")
        user, _ = login_user(self.db, "test@example.com", "password123")
        with self.assertRaises(AuthError) as ctx:
            delete_user(self.db, user.id, "wrongpassword")
        self.assertIn("Password is incorrect", str(ctx.exception))

    def test_email_case_insensitive(self):
        register_user(self.db, "Test@Example.COM", "testuser", "password123")
        user, session = login_user(self.db, "test@example.com", "password123")
        self.assertEqual(user.email, "test@example.com")


if __name__ == "__main__":
    unittest.main()