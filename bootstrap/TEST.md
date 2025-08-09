# Klarer Tests

While having unit tests for some bits and pieces of the different compiler stages
is good practice, real world tests are both, easier to write and reason about.

The tests in this file can be run with `md_tests.py`.

## Examples

Let's start with some examples to get a feel for the language and this test-suite.

### Hello, world!

```klarer
main = fun() do
    print("Hello, world!")
end
```

```
Hello, world!
```

## Primitives

### String

```klarer
main = fun() do
    print("PASS")
end
```

```
PASS
```

### Boolean

```klarer
main = fun() do
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
main = fun() do
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
main = fun() do
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
main = fun() do
    a = 'a'
    print(char_to_str(a))
end
```

```
a
```

### If Expressions

```klarer
main = fun() do
    if case true do
        print("PASS")
    end
end
```

```
PASS
```

**Multiple cases and else**

```klarer
main = fun() do
    print("middle one is taken:")
    if
        case false do print("FAIL")
        case true do print("PASS")
        else do print("FAIL")
    end

    print("else is taken:")
    if
        case false do print("FAIL")
        case false do print("FAIL")
        else do print("PASS")
    end
end
```

```
middle one is taken:
PASS
else is taken:
PASS
```

**Mutating variables that are defined outside the if**

```klarer
main = fun() do
    mut pass = "FAIL"
    mut num = 42

    if
        case false do
            pass = "FAIL!"
            num = 0
        case true do
            pass = "PASS"
        else do
            pass = "FAIL!!"
            num = 137
    end

    print(pass)
    print(int_to_str(num))
end
```

```
PASS
42
```

**Capturing the result of an if expression**

```klarer
main = fun() do
    s = if
        case false do "FAIL"
        case true do "PASS"
    end
    print(s)
end
```

```
PASS
```

**At least one if case**

```klarer
main = fun() do
    if else do end -- ERROR: Expected `case`, got `else`
end
```

## Mutability

### Mutable Variables

```klarer
main = fun() do
    mut s = "FAIL"
    s = "PASS"
    print(s)
end
```

```
PASS
```

**Mutable variables must be marked with `mut`**

```klarer
main = fun() do
    s = "FAIL"
    s = "PASS" -- ERROR: `s` is not mutable
end
```

### Mutable Function Parameters

> [!TODO]
> We need to actually mark the function parameter as mutable.
> For now, all non-primitive parameters are just mutable.

```klarer
f = fun(v) do
    v.value = "PASS"
end

main = fun() do
    mut v = {value = "FAIL"}
    f(v)
    print(v.value)
end
```

```
PASS
```

## Shape Inference

**Infer based on property access**

```klarer
deeply_nested = fun(o) do
    o.deeply.nested = 42
end

main = fun() do
    v = deeply_nested({deeply = {nested = "FAIL"}}) -- ERROR: `fun deeply_nested(o {deeply {nested Str}}) -> Unit` does not conform to shape `fun deeply_nested(o {deeply {nested Int}}) -> Unit`
end
```
