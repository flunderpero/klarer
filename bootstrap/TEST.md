# Klarer Tests

While having unit tests for some bits and pieces of the different compiler stages
is good practice, real world tests are both, easier to write and reason about.

The tests in this file can be run with `md_tests.py`.

## Examples

Let's start with some examples to get a feel for the language and this test-suite.

### Hello, world!

```klarer
main = fun():
    print("Hello, world!")
end
```

```
Hello, world!
```

## Primitives

### String

```klarer
main = fun():
    print("PASS")
end
```

```
PASS
```

### Boolean

```klarer
main = fun():
    t = true
    f = false
    print(bool_to_str(t))
    print(bool_to_str(f))
end
```

```
true
false
```

#### Boolean Expressions

```klarer
main = fun():
    print(bool_to_str(1 == 2))
    print(bool_to_str(1 == 1))
    print(bool_to_str(2 != 2))
    print(bool_to_str(1 != 2))
end
```

```
false
true
false
true
```

### Int

For now, we only support `Int` which is a 64-bit signed integer.

```klarer
main = fun():
    a = 42
    b = -42
    print(int_to_str(a))
    print(int_to_str(b))
end
```

```
42
-42
```

**Minimum and maximum values**

```klarer
main = fun():
    a = 9223372036854775807
    b = -9223372036854775808
    print(int_to_str(a))
    print(int_to_str(b))
end
```

```
9223372036854775807
-9223372036854775808
```

#### Arithmetic Expressions

```klarer
main = fun():
    a = 40 + 2
    print(int_to_str(a))

    b = 140 - 3
    print(int_to_str(b))

    c = 3 * 4
    print(int_to_str(c))

    d = 5 / 2
    print(int_to_str(d))
end
```

```
42
137
12
2
```

### Char

```klarer
main = fun():
    a = 'a'
    print(char_to_str(a))
end
```

```
a
```

### Lists

```klarer

Strings = {values [Str]}

main = fun():
    a = Strings{values = ["FAIL", "PASS"]}
    print(a.values[1])
end
```

```
PASS
```

**Lists of product shapes**

```klarer

Person = {name Str, age Int}

main = fun():
    list = [Person{name = "John", age = 42}, Person{name = "Jane", age = 24}]
    print(list[0].name)
    print(list[1].name)
end
```

```
John
Jane
```

**Lists of Int**

```klarer

main = fun():
    list = [42, 137]
    print(int_to_str(list[0]))
    print(int_to_str(list[1]))
end
```

```
42
137
```

**List of mixed shapes**

> [!TODO]
> We don't support sum shapes yet.

```todo

main = fun():
    list = []
    list = list + ["PASS"]
    -- `list` is `[Str]` here.
    print(list[0])

    list = list + [42]
    -- `list` is `[Str | Int]` here, that's why we have to `match`.
    match list[0]:
        case Str:
            print(list[0])
        case _:
            print("FAIL")
    end
    match list[1]:
        case Int:
            print(int_to_str(list[1]))
        case _:
            print("FAIL")
    end
end

```

```
PASS
42
```

#### List Operations

**Concatenating lists**

```klarer
main = fun():
    a = ["P", "A"]
    b = ["S"]
    c = a + ["S"] + b
    print(c[0])
    print(c[1])
    print(c[2])
    print(c[3])
end
```

```
P
A
S
S
```

## If Expressions

```klarer
main = fun():
    if case true:
        print("PASS")
    end
end
```

```
PASS
```

**Multiple cases and else**

```klarer
main = fun():
    print("middle one is taken:")
    if
        case false: print("FAIL")
        case true: print("PASS")
        else: print("FAIL")
    end

    print("else is taken:")
    if
        case false: print("FAIL")
        case false: print("FAIL")
        else: print("PASS")
    end
end
```

```
middle one is taken:
PASS
else is taken:
PASS
```

**Capturing the result of an if expression**

```klarer
main = fun():
    s = if
        case false: "FAIL"
        case true: "PASS"
    end
    print(s)
end
```

```
PASS
```

**At least one if case**

```klarer
main = fun():
    if else: end -- ERROR: Expected `case`, got `else`
end
```

## Assignment

**Shape assignment creates a copy**

```klarer
main = fun():
    a = {pass = "PASS"}
    b = {b = "b", a = a}

    print(a.pass)
    print(b.a.pass)
end
```

```
PASS
PASS
```

**Each assignment creates a new binding**

```klarer
main = fun():
    a = 42
    print(int_to_str(a))

    a = "PASS"
    print(a)

    a = {name = "John", age = 42}
    print(a.name)

    a = {value = "PASS"}
    print(a.value)
end
```

```
42
PASS
John
PASS
```

**Assignments are local to the scope they are created in.**

```klarer
main = fun():
    a = "PASS1"
    print(a)

    if case true:
        a = "PASS2"
        print(a)
    end

    print(a)
end
```

```
PASS1
PASS2
PASS1
```

## Behaviour

```klarer
@Value.print_value = fun(v {value Str}):
    print(v.value)
end

main = fun():
    v = {value = "PASS"} + @Value
    v.print_value()
end
```

```
PASS
```

**A shape literal should receive the behaviours of a shape alias**

```klarer
@Value.print_value = fun(v {value Str}):
    print(v.value)
end

Value = {value {}} + @Value

main = fun():
    v = Value{value = "PASS"}
    v.print_value()
end
```

```
PASS
```

### Interface Behaviours

```klarer
@MyToStr.to_str = fun(obj {}) Str

@Value.to_str = fun(obj {value Str}) Str:
    obj.value
end

Value = {value Str} + @Value

print_my_to_str = fun(v {} + @MyToStr):
    print(v.to_str())
end

main = fun():
    v = Value{value = "PASS"}
    print_my_to_str(v)
end
```

```
PASS
```

**Interface implementations must match the interface at call-site**

```klarer
@MyToStr.to_str = fun(obj {}) Str

-- Here we return an Int instead of a Str.
@Value.to_str = fun(obj {value Str}) Int:
    42
end

Value = {value Str} + @Value

print_my_to_str = fun(v {} + @MyToStr):
    print(v.to_str())
end

main = fun():
    v = Value{value = "PASS"}
    print_my_to_str(v) -- ERROR: `fun print_my_to_str(v {}) -> <unit>` cannot be called as `fun print_my_to_str(v Value{value Str}) -> <unit>`
end
```

**You cannot mix interface methods with non-interface methods in the same behaviour**

```klarer
@Value.foo = fun(): end

@Value.bar = fun() Str -- ERROR: Cannot add interface method `fun @Value.bar() -> Str` to non-interface behaviour `@Value`
```

```klarer
@Value.bar = fun() Str

@Value.foo = fun(): end -- ERROR: Cannot add method `fun @Value.foo() -> <unit>` to interface behaviour `@Value(interface)`.
```

## Forward Declarations

**Functions are forward declared**

```klarer

main = fun():
    print(pass("PASS"))
end

pass = fun(s Str) Str:
    s
end

```

```
PASS
```

**Shape aliases are forward declared**

```klarer

main = fun():
    p = Person{name = "John", age = 42}
    print_person_name(p)
end

-- We use the forward declared `Person` as a parameter shape.
print_person_name = fun(p Person):
    p.print_name()
end

-- We use the forward declared `Person` as a parameter shape of a behaviour method.
@Person.print_name = fun(p Person):
    print(p.name)
end

Person = {name Str, age Int} + @Person
```

```
John
```

**Forward declared shape aliases can be used in shape aliases**

```klarer
Container = {item Item, count Int}

Item = {value Str}

main = fun():
    c = Container{item = Item{value = "PASS"}, count = 1}
    print(c.item.value)
end
```

```
PASS
```

**Forward declared behaviours can be used in shape aliases and literals**

```klarer

main = fun():
    p = Person{name = "John", age = 42}
    p.print_name()

    p = {name = "Jane", age = 24} + @Person
    p.print_name()
end

Person = {name Str, age Int} + @Person

@Person.print_name = fun(p Person):
    print(p.name)
end

```

```
John
Jane
```

**Composing with forward declared shape aliases**

> [!TODO]
> We don't yet support `Base + ...` syntax. We need to remove `ast.ProductShape.composites`
> (and `.behaviours`) and introduce a new `ast.ShapeComp` node.

```todo
Combined = Base + {extra Int}

Base = {value Str}

main = fun():
    c = Combined{value = "PASS", extra = 42}
    print(c.value)
end
```

```
PASS
```

### Recursive Shapes

**Declaring a recursive shape alias**

> [!TODO]
> Until we have sum types, we cannot create a literal for a recursive shape alias.
> So for now, we just check that we can declare recursive shape aliases.

```klarer

Person = {name Str, age Int, parent Person}

main = fun():
end
```

**Declaring mutually recursive shape aliases**

```klarer

Address = {street Str, city Str, main Person}
Person = {name Str, age Int, address Address}

main = fun():
end
```

## Monomorphization

**Mutually recursive functions are monomorphized**

```klarer
foo = fun(n Int) Int:
    if
      case n == 0:
        42
      else:
        bar(n - 1)
    end
end

bar = fun(n Int) Int:
    foo(n)
end

main = fun():
    v = bar(2)
    print(int_to_str(v))
end
```

```
42
```
