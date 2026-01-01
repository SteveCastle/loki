package auth

import (
	"database/sql"
	"errors"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"golang.org/x/crypto/bcrypt"
)

var (
	ErrUserNotFound  = errors.New("user not found")
	ErrInvalidCreds  = errors.New("invalid credentials")
	ErrUserExists    = errors.New("username already exists")
	ErrSetupRequired = errors.New("setup required: please create a new user")
)

const DefaultAdminUsername = "admin"
const DefaultAdminPassword = "admin"

type User struct {
	ID           int64  `json:"id"`
	Username     string `json:"username"`
	PasswordHash string `json:"-"`
	CreatedAt    int64  `json:"created_at"`
}

type Claims struct {
	Username string `json:"username"`
	jwt.RegisteredClaims
}

type AuthService struct {
	db        *sql.DB
	jwtSecret []byte
}

func NewAuthService(db *sql.DB, secret string) *AuthService {
	return &AuthService{
		db:        db,
		jwtSecret: []byte(secret),
	}
}

// CreateDefaultUser creates a temporary admin user if no users exist
func (s *AuthService) CreateDefaultUser() error {
	var count int
	err := s.db.QueryRow("SELECT COUNT(*) FROM users").Scan(&count)
	if err != nil {
		return err
	}

	if count == 0 {
		return s.registerInternal(DefaultAdminUsername, DefaultAdminPassword)
	}
	return nil
}

// IsSetupRequired returns true if only the default admin user exists
// This means the user needs to create a real account
func (s *AuthService) IsSetupRequired() (bool, error) {
	var count int
	err := s.db.QueryRow("SELECT COUNT(*) FROM users").Scan(&count)
	if err != nil {
		return false, err
	}

	if count != 1 {
		return false, nil
	}

	// Check if the only user is the default admin
	var username string
	err = s.db.QueryRow("SELECT username FROM users LIMIT 1").Scan(&username)
	if err != nil {
		return false, err
	}

	return username == DefaultAdminUsername, nil
}

// HasDefaultAdminUser returns true if the default admin user still exists
func (s *AuthService) HasDefaultAdminUser() bool {
	var exists int
	err := s.db.QueryRow("SELECT 1 FROM users WHERE username = ?", DefaultAdminUsername).Scan(&exists)
	return err == nil
}

// DeleteDefaultAdmin removes the default admin user if it exists
func (s *AuthService) DeleteDefaultAdmin() error {
	_, err := s.db.Exec("DELETE FROM users WHERE username = ?", DefaultAdminUsername)
	return err
}

// registerInternal creates a user without deleting the default admin
func (s *AuthService) registerInternal(username, password string) error {
	// Check if user exists
	var exists int
	err := s.db.QueryRow("SELECT 1 FROM users WHERE username = ?", username).Scan(&exists)
	if err == nil {
		return ErrUserExists
	} else if err != sql.ErrNoRows {
		return err
	}

	// Hash password
	hash, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	if err != nil {
		return err
	}

	_, err = s.db.Exec("INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
		username, string(hash), time.Now().Unix())
	return err
}

// Register creates a new user and deletes the default admin if it exists
func (s *AuthService) Register(username, password string) error {
	// Don't allow registering with the default admin username
	if username == DefaultAdminUsername {
		return errors.New("cannot use reserved username 'admin'")
	}

	// Register the new user
	if err := s.registerInternal(username, password); err != nil {
		return err
	}

	// If the default admin user exists, delete it now that we have a real user
	if s.HasDefaultAdminUser() {
		if err := s.DeleteDefaultAdmin(); err != nil {
			// Log but don't fail - the user was created successfully
			// The admin will be cleaned up on next registration attempt
		}
	}

	return nil
}

func (s *AuthService) Login(username, password string) (string, error) {
	var user User
	err := s.db.QueryRow("SELECT id, username, password_hash FROM users WHERE username = ?", username).
		Scan(&user.ID, &user.Username, &user.PasswordHash)
	if err == sql.ErrNoRows {
		return "", ErrInvalidCreds
	} else if err != nil {
		return "", err
	}

	if err := bcrypt.CompareHashAndPassword([]byte(user.PasswordHash), []byte(password)); err != nil {
		return "", ErrInvalidCreds
	}

	// Generate Token
	expirationTime := time.Now().Add(24 * time.Hour * 365)
	claims := &Claims{
		Username: username,
		RegisteredClaims: jwt.RegisteredClaims{
			ExpiresAt: jwt.NewNumericDate(expirationTime),
		},
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	tokenString, err := token.SignedString(s.jwtSecret)
	if err != nil {
		return "", err
	}

	return tokenString, nil
}

func (s *AuthService) VerifyToken(tokenString string) (*Claims, error) {
	claims := &Claims{}
	token, err := jwt.ParseWithClaims(tokenString, claims, func(token *jwt.Token) (interface{}, error) {
		return s.jwtSecret, nil
	})

	if err != nil {
		return nil, err
	}

	if !token.Valid {
		return nil, errors.New("invalid token")
	}

	return claims, nil
}

func (s *AuthService) ListUsers() ([]User, error) {
	rows, err := s.db.Query("SELECT id, username, created_at FROM users ORDER BY username")
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var users []User
	for rows.Next() {
		var u User
		if err := rows.Scan(&u.ID, &u.Username, &u.CreatedAt); err != nil {
			return nil, err
		}
		users = append(users, u)
	}
	return users, nil
}

func (s *AuthService) DeleteUser(username string) error {
	// Prevent deleting the last user to avoid lockout
	var count int
	if err := s.db.QueryRow("SELECT COUNT(*) FROM users").Scan(&count); err != nil {
		return err
	}
	if count <= 1 {
		return errors.New("cannot delete the last user")
	}

	_, err := s.db.Exec("DELETE FROM users WHERE username = ?", username)
	return err
}
