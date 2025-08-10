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

### If Expressions

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

## Shape Inference

**Infer based on property access**

```klarer
deeply_nested = fun(o):
    o.deeply.nested == 42
end

main = fun():
    v = deeply_nested({deeply = {nested = "FAIL"}}) -- ERROR: `fun deeply_nested(o {deeply {nested Str}}) -> Bool` does not conform to shape `fun deeply_nested(o {deeply {nested Int}}) -> Bool`
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
