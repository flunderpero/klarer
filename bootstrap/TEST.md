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
