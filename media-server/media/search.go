package media

import (
	"fmt"
	"path/filepath"
	"strconv"
	"strings"
	"text/scanner"
	"unicode"
)

// TokenType represents the type of a token
type TokenType int

const (
	TokenError TokenType = iota
	TokenEOF
	TokenIdentifier // string, key
	TokenString     // "quoted string"
	TokenColon      // :
	TokenLParen     // (
	TokenRParen     // )
	TokenAnd        // AND
	TokenOr         // OR
	TokenNot        // NOT
	TokenGTE        // >=
	TokenLTE        // <=
	TokenGT         // >
	TokenLT         // <
)

// Token represents a lexical token
type Token struct {
	Type  TokenType
	Value string
}

// Lexer performs lexical analysis on the query string
type Lexer struct {
	scanner scanner.Scanner
	buf     Token // lookahead token
	hasBuf  bool
}

func NewLexer(input string) *Lexer {
	l := &Lexer{}
	l.scanner.Init(strings.NewReader(input))
	l.scanner.Mode = scanner.ScanIdents | scanner.ScanStrings | scanner.ScanInts | scanner.ScanFloats
	l.scanner.IsIdentRune = func(ch rune, i int) bool {
		return ch == '_' || ch == '-' || ch == '.' || ch == '*' || ch == '%' || unicode.IsLetter(ch) || unicode.IsDigit(ch)
	}
	return l
}

func (l *Lexer) scan() Token {
	if l.hasBuf {
		l.hasBuf = false
		return l.buf
	}

	tok := l.scanner.Scan()
	text := l.scanner.TokenText()

	switch tok {
	case scanner.EOF:
		return Token{Type: TokenEOF}
	case scanner.String:
		// Remove quotes
		return Token{Type: TokenString, Value: strings.Trim(text, "\"")}
	case '(':
		return Token{Type: TokenLParen, Value: "("}
	case ')':
		return Token{Type: TokenRParen, Value: ")"}
	case ':':
		return Token{Type: TokenColon, Value: ":"}
	case '>':
		if l.scanner.Peek() == '=' {
			l.scanner.Next()
			return Token{Type: TokenGTE, Value: ">="}
		}
		return Token{Type: TokenGT, Value: ">"}
	case '<':
		if l.scanner.Peek() == '=' {
			l.scanner.Next()
			return Token{Type: TokenLTE, Value: "<="}
		}
		return Token{Type: TokenLT, Value: "<"}
	case scanner.Ident, scanner.Int, scanner.Float:
		// Check for keywords
		upper := strings.ToUpper(text)
		switch upper {
		case "AND":
			return Token{Type: TokenAnd, Value: "AND"}
		case "OR":
			return Token{Type: TokenOr, Value: "OR"}
		case "NOT":
			return Token{Type: TokenNot, Value: "NOT"}
		}
		return Token{Type: TokenIdentifier, Value: text}
	default:
		// Handle special characters that scanner.Scan() returns as rune
		return Token{Type: TokenIdentifier, Value: text}
	}
}

func (l *Lexer) peek() Token {
	if !l.hasBuf {
		l.buf = l.scan()
		l.hasBuf = true
	}
	return l.buf
}

// Node is the interface for AST nodes
type Node interface {
	ToSQL() (string, []interface{})
	Evaluate(item MediaItem) bool
	HasExists() bool
}

// Parser parses tokens into an AST
type Parser struct {
	lexer *Lexer
}

func NewParser(input string) *Parser {
	return &Parser{lexer: NewLexer(input)}
}

func (p *Parser) Parse() (Node, error) {
	node, err := p.parseExpression()
	if err != nil {
		return nil, err
	}
	if p.lexer.peek().Type != TokenEOF {
		// If there's stuff left, implicitly assume AND if appropriate, or error
		// For robustness, let's treat trailing tokens as implicit AND with the previous expression
		// but checking for balanced parens is handled by recursive structure.
		// If we are here, we finished an expression but have tokens left.
		// Example: "tag:a tag:b" -> parseExpression parses "tag:a", stops. "tag:b" remains.
		// We should support implicit AND at the top level too.
		rhs, err := p.Parse()
		if err != nil {
			return nil, err
		}
		return &AndNode{Left: node, Right: rhs}, nil
	}
	return node, nil
}

// Expression -> Term { OR Term }
func (p *Parser) parseExpression() (Node, error) {
	left, err := p.parseTerm()
	if err != nil {
		return nil, err
	}

	for {
		if p.lexer.peek().Type == TokenOr {
			p.lexer.scan() // consume OR
			right, err := p.parseTerm()
			if err != nil {
				return nil, err
			}
			left = &OrNode{Left: left, Right: right}
		} else {
			break
		}
	}
	return left, nil
}

// Term -> Factor { [AND] Factor }
func (p *Parser) parseTerm() (Node, error) {
	left, err := p.parseFactor()
	if err != nil {
		return nil, err
	}

	for {
		peek := p.lexer.peek()
		if peek.Type == TokenAnd {
			p.lexer.scan() // consume AND
			right, err := p.parseFactor()
			if err != nil {
				return nil, err
			}
			left = &AndNode{Left: left, Right: right}
		} else if peek.Type == TokenNot || peek.Type == TokenIdentifier || peek.Type == TokenString || peek.Type == TokenLParen {
			// Implicit AND
			right, err := p.parseFactor()
			if err != nil {
				return nil, err
			}
			left = &AndNode{Left: left, Right: right}
		} else {
			break
		}
	}
	return left, nil
}

// Factor -> NOT Factor | ( Expression ) | Predicate
func (p *Parser) parseFactor() (Node, error) {
	token := p.lexer.peek()

	if token.Type == TokenNot {
		p.lexer.scan() // consume NOT
		child, err := p.parseFactor()
		if err != nil {
			return nil, err
		}
		return &NotNode{Child: child}, nil
	}

	if token.Type == TokenLParen {
		p.lexer.scan() // consume (
		expr, err := p.parseExpression()
		if err != nil {
			return nil, err
		}
		if p.lexer.peek().Type != TokenRParen {
			return nil, fmt.Errorf("expected closing parenthesis")
		}
		p.lexer.scan() // consume )
		return expr, nil
	}

	return p.parsePredicate()
}

// Predicate -> key : value | key operator value
func (p *Parser) parsePredicate() (Node, error) {
	keyToken := p.lexer.scan()
	if keyToken.Type != TokenIdentifier {
		return nil, fmt.Errorf("expected identifier, got %v", keyToken)
	}

	// Check for operator
	opToken := p.lexer.peek()
	var operator string

	switch opToken.Type {
	case TokenColon:
		p.lexer.scan() // consume :
		operator = "="
	case TokenGT:
		p.lexer.scan()
		operator = ">"
	case TokenLT:
		p.lexer.scan()
		operator = "<"
	case TokenGTE:
		p.lexer.scan()
		operator = ">="
	case TokenLTE:
		p.lexer.scan()
		operator = "<="
	default:
		// Default to text search on description/path/tags if no operator?
		// But existing logic is strict on key:value.
		// Let's assume standard key:value syntax.
		return nil, fmt.Errorf("expected ':', '>', '<', '>=', '<=' after key, got %v", opToken.Value)
	}

	// Value
	valToken := p.lexer.scan()
	if valToken.Type != TokenIdentifier && valToken.Type != TokenString {
		// Allow identifiers, strings, numbers
		// Note: TokenInt/Float are not explicitly in our enum but scanner returns them
		// Our scan() maps everything else to TokenIdentifier so this check is simplified
	}

	// Handle wildcards in value for standard equality
	value := valToken.Value
	if operator == "=" && (strings.Contains(value, "*") || strings.Contains(value, "%")) {
		operator = "LIKE"
		value = strings.ReplaceAll(value, "*", "%")
	}

	return &ConditionNode{
		Column:   strings.ToLower(keyToken.Value),
		Operator: operator,
		Value:    value,
	}, nil
}

// TokenInt isn't defined in enum, reusing TokenIdentifier for values as scanner simplifies it
// Correcting logic in parsePredicate: lexer.scan() handles the types.

// --- AST Implementations ---

type ConditionNode struct {
	Column   string
	Operator string
	Value    string
}

func (n *ConditionNode) ToSQL() (string, []interface{}) {
	column := n.Column
	op := n.Operator
	val := n.Value

	switch column {
	case "path":
		return "m.path " + op + " ?", []interface{}{val}
	case "description":
		return "m.description " + op + " ?", []interface{}{val}
	case "size":
		// Handle size conversion if needed? Assuming bytes
		return "m.size " + op + " ?", []interface{}{val}
	case "hash":
		return "m.hash " + op + " ?", []interface{}{val}
	case "width":
		return "m.width " + op + " ?", []interface{}{val}
	case "height":
		return "m.height " + op + " ?", []interface{}{val}
	case "tags":
		if strings.ToLower(val) == "none" {
			// tags:none
			return "NOT EXISTS (SELECT 1 FROM media_tag_by_category mtbc WHERE mtbc.media_path = m.path)", nil
		}
		// Unknown tags value
		return "1=0", nil
	case "pathdir":
		// Normalized path handling
		pathSep := "/"
		if strings.Contains(val, "\\") {
			pathSep = "\\"
		}
		if !strings.HasSuffix(val, pathSep) {
			val += pathSep
		}
		// LIKE 'dir/%' AND NOT LIKE 'dir/%/%'
		return "(m.path LIKE ? AND m.path NOT LIKE ?)", []interface{}{val + "%", val + "%" + pathSep + "%"}
	case "tagcount":
		// (SELECT COUNT(*) ...) op ?
		return "(SELECT COUNT(*) FROM media_tag_by_category mtbc WHERE mtbc.media_path = m.path) " + op + " ?", []interface{}{val}
	case "tag":
		// EXISTS (...)
		return fmt.Sprintf("EXISTS (SELECT 1 FROM media_tag_by_category mtbc WHERE mtbc.media_path = m.path AND mtbc.tag_label %s ?)", op), []interface{}{val}
	case "category":
		return fmt.Sprintf("EXISTS (SELECT 1 FROM media_tag_by_category mtbc WHERE mtbc.media_path = m.path AND mtbc.category_label %s ?)", op), []interface{}{val}
	case "exists":
		// Handled in Go, always true in SQL to fetch candidate
		return "1=1", nil
	default:
		// Ignore unknown columns or fail?
		return "1=1", nil // Ignore safely
	}
}

func (n *ConditionNode) Evaluate(item MediaItem) bool {
	switch n.Column {
	case "exists":
		expected, _ := strconv.ParseBool(n.Value)
		return item.Exists == expected
	case "path":
		return compareString(item.Path, n.Operator, n.Value)
	case "description":
		if !item.Description.Valid {
			return false
		}
		return compareString(item.Description.String, n.Operator, n.Value)
	case "size":
		if !item.Size.Valid {
			return false
		}
		v, _ := strconv.ParseInt(n.Value, 10, 64)
		return compareInt(item.Size.Int64, n.Operator, v)
	case "width":
		if !item.Width.Valid {
			return false
		}
		v, _ := strconv.ParseInt(n.Value, 10, 64)
		return compareInt(item.Width.Int64, n.Operator, v)
	case "height":
		if !item.Height.Valid {
			return false
		}
		v, _ := strconv.ParseInt(n.Value, 10, 64)
		return compareInt(item.Height.Int64, n.Operator, v)
	case "tag":
		// item.Tags contains all tags
		for _, t := range item.Tags {
			if compareString(t.Label, n.Operator, n.Value) {
				return true
			}
		}
		return false
	case "category":
		for _, t := range item.Tags {
			if compareString(t.Category, n.Operator, n.Value) {
				return true
			}
		}
		return false
	case "tags":
		if strings.ToLower(n.Value) == "none" {
			return len(item.Tags) == 0
		}
		return false
	case "tagcount":
		v, _ := strconv.ParseInt(n.Value, 10, 64)
		return compareInt(int64(len(item.Tags)), n.Operator, v)
	case "pathdir":
		dir := filepath.Dir(item.Path)
		target := filepath.Clean(n.Value)
		// Simple check: is dir equal to target
		// Note: The SQL does strictly one level.
		// Go's filepath.Dir gives the parent.
		return strings.EqualFold(dir, target)
	default:
		return true
	}
}

func (n *ConditionNode) HasExists() bool {
	return n.Column == "exists"
}

// Helper functions for comparison
func compareString(val, op, target string) bool {
	val = strings.ToLower(val)
	target = strings.ToLower(target)
	switch op {
	case "=":
		return val == target
	case "LIKE":
		// Simple glob matching
		match, _ := filepath.Match(target, val)
		return match
	default:
		return val == target
	}
}

func compareInt(val int64, op string, target int64) bool {
	switch op {
	case "=":
		return val == target
	case ">":
		return val > target
	case "<":
		return val < target
	case ">=":
		return val >= target
	case "<=":
		return val <= target
	default:
		return false
	}
}

type AndNode struct {
	Left, Right Node
}

func (n *AndNode) ToSQL() (string, []interface{}) {
	lSql, lArgs := n.Left.ToSQL()
	rSql, rArgs := n.Right.ToSQL()
	return "(" + lSql + " AND " + rSql + ")", append(lArgs, rArgs...)
}

func (n *AndNode) Evaluate(item MediaItem) bool {
	return n.Left.Evaluate(item) && n.Right.Evaluate(item)
}

func (n *AndNode) HasExists() bool {
	return n.Left.HasExists() || n.Right.HasExists()
}

type OrNode struct {
	Left, Right Node
}

func (n *OrNode) ToSQL() (string, []interface{}) {
	lSql, lArgs := n.Left.ToSQL()
	rSql, rArgs := n.Right.ToSQL()
	return "(" + lSql + " OR " + rSql + ")", append(lArgs, rArgs...)
}

func (n *OrNode) Evaluate(item MediaItem) bool {
	return n.Left.Evaluate(item) || n.Right.Evaluate(item)
}

func (n *OrNode) HasExists() bool {
	return n.Left.HasExists() || n.Right.HasExists()
}

type NotNode struct {
	Child Node
}

func (n *NotNode) ToSQL() (string, []interface{}) {
	cSql, cArgs := n.Child.ToSQL()
	return "NOT (" + cSql + ")", cArgs
}

func (n *NotNode) Evaluate(item MediaItem) bool {
	return !n.Child.Evaluate(item)
}

func (n *NotNode) HasExists() bool {
	return n.Child.HasExists()
}
