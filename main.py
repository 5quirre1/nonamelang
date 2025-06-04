import re
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
@dataclass
class Token:
    type: str
    value: str
    line: int = 0
class Lexer:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.line = 1
    def tokenize(self) -> List[Token]:
        tokens = []
        while self.pos < len(self.text):
            if self.current_char().isspace():
                if self.current_char() == '\n':
                    self.line += 1
                self.pos += 1
            elif self.current_char() == '/' and self.peek() == '/':
                self.skip_comment()
            elif self.current_char().isdigit():
                tokens.append(self.number())
            elif self.current_char().isalpha() or self.current_char() == '_':
                tokens.append(self.identifier())
            elif self.current_char() == '"':
                tokens.append(self.string())
            else:
                tokens.append(self.operator())
        return tokens
    def current_char(self) -> str:
        return self.text[self.pos] if self.pos < len(self.text) else ''
    def peek(self, offset: int = 1) -> str:
        pos = self.pos + offset
        return self.text[pos] if pos < len(self.text) else ''
    def skip_comment(self):
        while self.current_char() and self.current_char() != '\n':
            self.pos += 1
    def number(self) -> Token:
        start = self.pos
        while self.current_char().isdigit() or self.current_char() == '.':
            self.pos += 1
        return Token('NUMBER', self.text[start:self.pos], self.line)
    def identifier(self) -> Token:
        start = self.pos
        while self.current_char().isalnum() or self.current_char() == '_':
            self.pos += 1
        value = self.text[start:self.pos]
        keywords = {'let', 'fn', 'if', 'else', 'while', 'return', 'print', 'true', 'false'}
        token_type = 'KEYWORD' if value in keywords else 'IDENTIFIER'
        return Token(token_type, value, self.line)
    def string(self) -> Token:
        self.pos += 1
        start = self.pos
        while self.current_char() and self.current_char() != '"':
            self.pos += 1
        value = self.text[start:self.pos]
        self.pos += 1
        return Token('STRING', value, self.line)
    def operator(self) -> Token:
        char = self.current_char()
        self.pos += 1
        if char == '=' and self.current_char() == '=':
            self.pos += 1
            return Token('EQ', '==', self.line)
        elif char == '!' and self.current_char() == '=':
            self.pos += 1
            return Token('NE', '!=', self.line)
        elif char == '<' and self.current_char() == '=':
            self.pos += 1
            return Token('LE', '<=', self.line)
        elif char == '>' and self.current_char() == '=':
            self.pos += 1
            return Token('GE', '>=', self.line)
        ops = {
            '+': 'PLUS', '-': 'MINUS', '*': 'MUL', '/': 'DIV',
            '=': 'ASSIGN', '<': 'LT', '>': 'GT',
            '(': 'LPAREN', ')': 'RPAREN', '{': 'LBRACE', '}': 'RBRACE',
            ',': 'COMMA', ';': 'SEMICOLON'
        }
        return Token(ops.get(char, 'UNKNOWN'), char, self.line)
class ASTNode:
    pass
@dataclass
class NumberNode(ASTNode):
    value: float
@dataclass
class StringNode(ASTNode):
    value: str
@dataclass
class BoolNode(ASTNode):
    value: bool
@dataclass
class IdentifierNode(ASTNode):
    name: str
@dataclass
class BinaryOpNode(ASTNode):
    left: ASTNode
    op: str
    right: ASTNode
@dataclass
class AssignNode(ASTNode):
    name: str
    value: ASTNode
@dataclass
class CallNode(ASTNode):
    name: str
    args: List[ASTNode]
@dataclass
class FunctionNode(ASTNode):
    name: str
    params: List[str]
    body: List[ASTNode]
@dataclass
class IfNode(ASTNode):
    condition: ASTNode
    then_body: List[ASTNode]
    else_body: Optional[List[ASTNode]] = None
@dataclass
class WhileNode(ASTNode):
    condition: ASTNode
    body: List[ASTNode]
@dataclass
class ReturnNode(ASTNode):
    value: Optional[ASTNode] = None
@dataclass
class PrintNode(ASTNode):
    value: ASTNode
class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
    def current_token(self) -> Optional[Token]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None
    def consume(self, expected_type: str = None) -> Token:
        token = self.current_token()
        if expected_type and (not token or token.type != expected_type):
            raise SyntaxError(f"expected {expected_type}, got {token.type if token else 'EOF'}")
        self.pos += 1
        return token
    def parse(self) -> List[ASTNode]:
        statements = []
        while self.current_token():
            stmt = self.statement()
            if stmt:
                statements.append(stmt)
        return statements
    def statement(self) -> Optional[ASTNode]:
        token = self.current_token()
        if not token:
            return None
        if token.value == 'let':
            return self.assignment()
        elif token.value == 'fn':
            return self.function()
        elif token.value == 'if':
            return self.if_statement()
        elif token.value == 'while':
            return self.while_statement()
        elif token.value == 'return':
            return self.return_statement()
        elif token.value == 'print':
            return self.print_statement()
        else:
            expr = self.expression()
            if self.current_token() and self.current_token().type == 'SEMICOLON':
                self.consume('SEMICOLON')
            return expr
    def assignment(self) -> AssignNode:
        self.consume('KEYWORD')
        name = self.consume('IDENTIFIER').value
        self.consume('ASSIGN')
        value = self.expression()
        if self.current_token() and self.current_token().type == 'SEMICOLON':
            self.consume('SEMICOLON')
        return AssignNode(name, value)
    def function(self) -> FunctionNode:
        self.consume('KEYWORD')
        name = self.consume('IDENTIFIER').value
        self.consume('LPAREN')
        params = []
        while self.current_token() and self.current_token().type != 'RPAREN':
            params.append(self.consume('IDENTIFIER').value)
            if self.current_token() and self.current_token().type == 'COMMA':
                self.consume('COMMA')
        self.consume('RPAREN')
        self.consume('LBRACE')
        body = []
        while self.current_token() and self.current_token().type != 'RBRACE':
            stmt = self.statement()
            if stmt:
                body.append(stmt)
        self.consume('RBRACE')
        return FunctionNode(name, params, body)
    def if_statement(self) -> IfNode:
        self.consume('KEYWORD')
        self.consume('LPAREN')
        condition = self.expression()
        self.consume('RPAREN')
        self.consume('LBRACE')
        then_body = []
        while self.current_token() and self.current_token().type != 'RBRACE':
            stmt = self.statement()
            if stmt:
                then_body.append(stmt)
        self.consume('RBRACE')
        else_body = None
        if self.current_token() and self.current_token().value == 'else':
            self.consume('KEYWORD')
            self.consume('LBRACE')
            else_body = []
            while self.current_token() and self.current_token().type != 'RBRACE':
                stmt = self.statement()
                if stmt:
                    else_body.append(stmt)
            self.consume('RBRACE')
        return IfNode(condition, then_body, else_body)
    def while_statement(self) -> WhileNode:
        self.consume('KEYWORD')
        self.consume('LPAREN')
        condition = self.expression()
        self.consume('RPAREN')
        self.consume('LBRACE')
        body = []
        while self.current_token() and self.current_token().type != 'RBRACE':
            stmt = self.statement()
            if stmt:
                body.append(stmt)
        self.consume('RBRACE')
        return WhileNode(condition, body)
    def return_statement(self) -> ReturnNode:
        self.consume('KEYWORD')
        value = None
        if (self.current_token() and
            self.current_token().type not in ['SEMICOLON', 'RBRACE']):
            value = self.expression()
        if self.current_token() and self.current_token().type == 'SEMICOLON':
            self.consume('SEMICOLON')
        return ReturnNode(value)
    def print_statement(self) -> PrintNode:
        self.consume('KEYWORD')
        self.consume('LPAREN')
        value = self.expression()
        self.consume('RPAREN')
        if self.current_token() and self.current_token().type == 'SEMICOLON':
            self.consume('SEMICOLON')
        return PrintNode(value)
    def expression(self) -> ASTNode:
        return self.comparison()
    def comparison(self) -> ASTNode:
        left = self.arithmetic()
        while (self.current_token() and
               self.current_token().type in ['EQ', 'NE', 'LT', 'GT', 'LE', 'GE']):
            op = self.consume().value
            right = self.arithmetic()
            left = BinaryOpNode(left, op, right)
        return left
    def arithmetic(self) -> ASTNode:
        left = self.term()
        while (self.current_token() and
               self.current_token().type in ['PLUS', 'MINUS']):
            op = self.consume().value
            right = self.term()
            left = BinaryOpNode(left, op, right)
        return left
    def term(self) -> ASTNode:
        left = self.factor()
        while (self.current_token() and
               self.current_token().type in ['MUL', 'DIV']):
            op = self.consume().value
            right = self.factor()
            left = BinaryOpNode(left, op, right)
        return left
    def factor(self) -> ASTNode:
        token = self.current_token()
        if token.type == 'NUMBER':
            self.consume()
            return NumberNode(float(token.value))
        elif token.type == 'STRING':
            self.consume()
            return StringNode(token.value)
        elif token.type == 'KEYWORD' and token.value in ['true', 'false']:
            self.consume()
            return BoolNode(token.value == 'true')
        elif token.type == 'IDENTIFIER':
            name = self.consume().value
            if self.current_token() and self.current_token().type == 'LPAREN':
                self.consume('LPAREN')
                args = []
                while self.current_token() and self.current_token().type != 'RPAREN':
                    args.append(self.expression())
                    if self.current_token() and self.current_token().type == 'COMMA':
                        self.consume('COMMA')
                self.consume('RPAREN')
                return CallNode(name, args)
            else:
                return IdentifierNode(name)
        elif token.type == 'LPAREN':
            self.consume('LPAREN')
            expr = self.expression()
            self.consume('RPAREN')
            return expr
        else:
            raise SyntaxError(f"unexpected token: {token.type}")
class Environment:
    def __init__(self, parent: Optional['Environment'] = None):
        self.vars: Dict[str, Any] = {}
        self.parent = parent
    def get(self, name: str) -> Any:
        if name in self.vars:
            return self.vars[name]
        elif self.parent:
            return self.parent.get(name)
        else:
            raise NameError(f"undefined variable: {name}")
    def set(self, name: str, value: Any):
        self.vars[name] = value
class Function:
    def __init__(self, params: List[str], body: List[ASTNode], closure: Environment):
        self.params = params
        self.body = body
        self.closure = closure
class ReturnException(Exception):
    def __init__(self, value: Any):
        self.value = value
class Interpreter:
    def __init__(self):
        self.global_env = Environment()
    def interpret(self, ast: List[ASTNode]) -> Any:
        result = None
        for node in ast:
            result = self.evaluate(node, self.global_env)
        return result
    def evaluate(self, node: ASTNode, env: Environment) -> Any:
        if isinstance(node, NumberNode):
            return node.value
        elif isinstance(node, StringNode):
            return node.value
        elif isinstance(node, BoolNode):
            return node.value
        elif isinstance(node, IdentifierNode):
            return env.get(node.name)
        elif isinstance(node, BinaryOpNode):
            left = self.evaluate(node.left, env)
            right = self.evaluate(node.right, env)
            if node.op == '+':
                return left + right
            elif node.op == '-':
                return left - right
            elif node.op == '*':
                return left * right
            elif node.op == '/':
                return left / right
            elif node.op == '==':
                return left == right
            elif node.op == '!=':
                return left != right
            elif node.op == '<':
                return left < right
            elif node.op == '>':
                return left > right
            elif node.op == '<=':
                return left <= right
            elif node.op == '>=':
                return left >= right
        elif isinstance(node, AssignNode):
            value = self.evaluate(node.value, env)
            env.set(node.name, value)
            return value
        elif isinstance(node, CallNode):
            func = env.get(node.name)
            if isinstance(func, Function):
                args = [self.evaluate(arg, env) for arg in node.args]
                return self.call_function(func, args)
            else:
                raise TypeError(f"{node.name} is not a function")
        elif isinstance(node, FunctionNode):
            func = Function(node.params, node.body, env)
            env.set(node.name, func)
            return func
        elif isinstance(node, IfNode):
            condition = self.evaluate(node.condition, env)
            if condition:
                return self.execute_block(node.then_body, env)
            elif node.else_body:
                return self.execute_block(node.else_body, env)
        elif isinstance(node, WhileNode):
            result = None
            while self.evaluate(node.condition, env):
                result = self.execute_block(node.body, env)
            return result
        elif isinstance(node, ReturnNode):
            value = self.evaluate(node.value, env) if node.value else None
            raise ReturnException(value)
        elif isinstance(node, PrintNode):
            value = self.evaluate(node.value, env)
            print(value)
            return value
        return None
    def execute_block(self, statements: List[ASTNode], env: Environment) -> Any:
        result = None
        for stmt in statements:
            result = self.evaluate(stmt, env)
        return result
    def call_function(self, func: Function, args: List[Any]) -> Any:
        if len(args) != len(func.params):
            raise TypeError(f"function expects {len(func.params)} args, got {len(args)}")
        func_env = Environment(func.closure)
        for param, arg in zip(func.params, args):
            func_env.set(param, arg)
        try:
            return self.execute_block(func.body, func_env)
        except ReturnException as ret:
            return ret.value
def runyoustupidnonswagstupidlanguagethatnoonewilluselmfaololhahahhaathislangdoesntevenhaveaname(code: str):
    try:
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        interpreter = Interpreter()
        return interpreter.interpret(ast)
    except Exception as e:
        print(f"error: {e}")
        return None
if __name__ == "__main__":
    test1 = """
    let x = 10;
    let y = 20;
    let sum = x + y;
    print(sum);
    """
    test2 = """
    fn greet(name) {
        return "hai " + name
    }
    print(greet("greg"));
    """
    test3 = """
    let i = 1;
    while (i <= 5) {
        print(i);
        let i = i + 1;
    }
    """
    print("=== Test 1: basic math ===")
    runyoustupidnonswagstupidlanguagethatnoonewilluselmfaololhahahhaathislangdoesntevenhaveaname(test1)
    print("\n=== Test 2: greet ===")
    runyoustupidnonswagstupidlanguagethatnoonewilluselmfaololhahahhaathislangdoesntevenhaveaname(test2)
    print("\n=== Test 3: while loop ===")
    runyoustupidnonswagstupidlanguagethatnoonewilluselmfaololhahahhaathislangdoesntevenhaveaname(test3)
