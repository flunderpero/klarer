# Klarer - Don't Fight the Compiler

## Introduction

Klarer tries to put back the fun in programming.

## Structural (not) Types?

Klarer uses structural _Shapes_ instead of nominal types.

Two shapes are structurally equal if they have the same attributes, the same variants, and the same
behaviours.

The shape of a function is determined by the use of its parameters and its result.

```klarer

get_value = fun(o):
    o.value
end

main = fun():
    str_value = {value = "PASS"}
    print(get_value(str_value))

    int_value = {value = 42, text = "fourty two"}
    print(int_to_str(get_value(int_value)))
end

```

```
PASS
42
```

In this example, the signature of `get_value` is inferred to be `fun(o {value {}})` where `o` is
said to be _"anything with a `value` attribute"_, where the value attribute is the empty shape `{}`.
The empty shape `{}` is subsumed by any other shape, i.e. any shape conforms to the empty shape.

The return shape of `get_value` is inferred to be the same as the parameter shape `o`.

In the `main` function, we create two shape literals, `str_value` and `int_value` with both
conforming to the parameter `o` of `get_value`.

The compiler will monomorphize `get_value` into two concrete functions, one that operates on the
shape of `str_value` and one that operates on the shape of `int_value`.
