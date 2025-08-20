# Klarer - Don't Fight the Compiler

## Introduction

Klarer tries to put back the fun in programming.

## Shapes, Not Types

Klarer uses structural _shapes_ instead of nominal types.

You _describe_ a value, its fields and behaviour, instead of giving it a _type_.

```klarer

print_value = fun(obj {value Str}):
    print(obj.value)
end

main = fun():
    v = {value = "PASS", extra = 42}
    print_value(v)
end
```

```
PASS
```

Here, `print_value` will accept any shape that has _at least_ a `value` field that is a `Str`.
When we call `print_value` we pass it a shape literal that has more fields, but that is fine.

## Polymorphism

Klarer has no concept of _parametric polymorphism_ or _generics_. Due to the structural nature of
shapes, _everything_ is polymorphic.

```klarer

get_value = fun(obj {value {}}) {}:
    obj.value
end

main = fun():
    str_value = {value = "PASS"}
    print(get_value(str_value))

    int_value = {value = 42, text = "forty two"}
    print(get_value(int_value))
end

```

```
PASS
42
```

In this example, the signature of `get_value` is `fun(obj {value {}}) {}` where `obj` can
be anything with _at least_ a `value` field that can be of any shape.
The `{}` in `{value {}}` and the return shape `{}` are the _empty shape_. Every shape conforms to
the empty shape `{}`, because the empty shape does not require any fields or
behaviours. We will learn about behaviours later.

### How Polymorphism Works (Early Monomorphization)

In `main`, we see how we can call `get_value` with different _shape literals_, i.e. `str_value`
and `int_value`. We can do this, because those literals conform to the shape of the `obj`
parameter of `get_value` - they have _at least_ a `value` field that can be anything.

But how does the return shape of `get_value` match the shape of `obj.value`? We did not
specify a relationship between `obj.value` and the return shape.

This has something to do with how the compiler evaluates function calls. When a function is called,
the compiler will actually substitute all parameter shapes with their concrete argument shapes. So
for the first call to `get_value`, the compiler will substitute the parameter shape `{value {}}` with
the actual argument shape `{value Str}` and then evaluate the body of `get_value(obj {value Str})`.
It will then discover that the _actual_ return shape for _this_ call is `Str`, because we
return `obj.value` which is a `Str` in _this_ call.
So we can directly print the result. (`print` only works on `Str` values.)

### Shapes Are Not Types

In Klarer, don't think of shapes as types. Shapes are not (nominal) types. In nominal type systems,
you declare a type by giving it a name and then declaring the operations that are valid on that type.
Example:

```go
package main

import "fmt"

type Person struct {
    Name string
    Age int
}

type Employee struct {
    Name string
    Age int
}

func print_name(p Person) {
    fmt.Println(p.Name)
}

func main() {
    p := Person{Name: "John", Age: 42}
    print_name(p)

    e := Employee{Name: "Jane", Age: 24}
    print_name(e)
}
```

This program will not compile because you cannot call `print_name` with an `Employee`.

In Klarer, shapes are not types. Shapes are the same if they have the same fields and behaviours,
i.e. if they look the same and behave the same, then they are the same shape.
Furthermore, a shape A can be substituted for a shape B if A _conforms to_ B. That is, A has _at
least_ all the fields and behaviours of B but it can have more fields and behaviours. This is why
polymorphism works "out of the box" in Klarer.

```klarer

-- These are not a type declarations.
-- They are merely aliases for shape definitions.
Person = {name Str, age Int}
Employee = {name Str, age Int}

print_name = fun(p Person):
    print(p.name)
end

main = fun():
    p = Person{name = "John", age = 42}
    print_name(p)

    e = Employee{name = "Jane", age = 24}
    print_name(e)

    -- We only need to conform to the shape of `p` to call `print_name`.
    other = {name = "Jake", age = 31}
    print_name(other)
end
```

```
John
Jane
Jake
```

Because `Employee` and `Person` are the exact same shape, they conform to each other and can be
used interchangeably.

The compiler does not work on shape names (aka aliases) at all. Shapes are always fully expanded.
So the compiler sees `print_name` as `fun(p {name Str, age Int})` and both `Person` and `Employee`
as `{name Str, age Int}`. The compiler only checks whether the shape of the function _argument_
conforms to the shape of the function _parameter_, i.e. it has _at least_ all the fields of
the parameter shape.

#### How To Distinguish Shapes Then?

But sometimes, you want to distinguish between two shapes. You can do so using _atoms_. We will
discuss atoms later.

### How To Express Relationships Between Shapes?

In languages with nominal type systems and generics, you explicitly declare type relationships.
In Klarer, these relationships emerge naturally from usage.

One of the most common use cases that require declaring relationships between types in languages
with a nominal type system, is higher-order functions (HOFs). Example:

```go

package main

import "strconv"

func Map[I any, O any](in I, f func(I) O) O {
	return f(in)
}

func main() {
	s := Map[int, string](42, func(in int) string {
		return strconv.Itoa(in)
	})
	print(s)
}

```

How do you write this in Klarer?

```klarer

map = fun(in {}, f fun(in {}) {}) {}:
    f(in)
end

itoa = fun(in Int) Str:
    in.to_str()
end

main = fun():
    s = map(42, itoa)
    print(s)
end
```

```
42
```

Again, when `map` is called, the compiler substitutes the signature of `map` with the
parameter shape `map = fun({in Int, f fun(in Int) Str}) {}`. It will then type check the
body of `map` again and substitute the return shape `{}` with the actual return shape `Str`.

### Constraints

In Klarer, constraints aren't a special feature - they're just shapes with more specific
requirements. When you write `{value Str}` instead of `{}`, you're saying "I need at least
a `value` field that is a `Str`." This naturally creates a constraint without any special syntax.

The beauty is that constraints compose naturally with Klarer's conformance rules. Any shape
that has the required fields (and possibly more) will work:

```klarer

-- We expect `in` to have _at least_ a `value` field.
map_value = fun(in {value {}}, f fun(in {}) {}) {}:
    f(in.value)
end

itoa = fun(in Int) Str:
    in.to_str()
end

main = fun():
    -- `map_value` expects `in` to have _at least_ a `value` field.
    -- But it does not care about the `extra` field.
    s = map_value({value = 42, extra = "extra"}, itoa)
    print(s)

    -- This would not compile because `in` does not have a `value` field.
    -- s = map_value({extra = "extra"}, itoa)

    -- This would not compile because the compiler detects that `itoa` cannot
    -- be called with `Str` (coming from `{value Str}`).
    -- s = map_value({value = "Hello"}, itoa)
end

```

```
42
```

### Kinds of Shapes

Klarer has these kinds of shapes:

#### Primitive Shapes: `Bool`, `Char`, `Int`, `Str`

Primitive shapes are built-in. A primitive shape conforms only to itself and the empty shape (`{}`).

Primitive shapes have built-in behaviours.

#### Product Shapes: `{name Str, age Int}`

Product shapes are shapes with fields. A product shape only conforms to other product shapes if
it has at least all the fields of the other shape and its behaviours conform to the other shape's
behaviours.

Product shapes can have behaviours attached to them.

#### Sum Shapes: `Str | Int | {value Str}`

Sum shapes are shapes with variants. A sum shape only conforms to other sum shapes if it has at
least all the variants of the other shape and its behaviours conform to the other shape's
behaviours.

Sum shapes also conform to the empty shape (`{}`).

Sum shapes cannot have behaviours attached to them. Behaviour methods emerge naturally if all
variants of a sum shape have the same behaviour method.

#### Function Shapes: `fun(in Str) Int`

Function shapes are shapes with parameters and result. A function shape only conforms to other
function shapes if all its parameters and result conform to the other function's parameters and
result.

Function shapes also conform to the empty shape (`{}`).

Function shapes _cannot_ have behaviours attached to them.

### Values

Klarer has these kinds of values:

#### Primitive Values

```klarer

main = fun():
    -- Bool
    print(true)
    print(false)

    -- Char
    print('c')

    -- Int
    print(42)

    -- Str
    print("PASS")

```

```
true
false
c
42
PASS
```

#### Product Values

A _product value_ is either created with an anonymous product shape or with a named product shape.

```klarer

Person = {name Str, age Int}

main = fun():
    -- Anonymous product shape.
    anon = {name = "John", age = 42}
    print(anon.name)

    -- Named product shape.
    person = Person{name = "Jane", age = 24}
    print(person.name)
end

```

```
John
Jane
```

Creating a product value based on a named product shape checks that all fields are present and
conform to their shapes.

```klarer

Person = {name Str, age Int}

main = fun():
    p = Person{name = "John"} -- ERROR: `{name Str}` does not conform to shape `Person`
end

```

```klarer

Person = {name Str, age Int}

main = fun():
    p = Person{name = "John", age = true} -- ERROR: `{name Str, age Bool}` does not conform to shape `Person`
end

```

All behaviours attached to a named product shape are also attached to the product.

```klarer

@Person.print_name = fun(p {name Str}):
    print(p.name)
end

Person = {name Str, age Int} + @Person

main = fun():
    p = Person{name = "John", age = 42}
    -- The behaviour function `@Person.print_name` is attached to the value `p`.
    p.print_name()
end

```

```
John
```

#### Sum Values

_Sum values_ emerge naturally from usage.

```todo

main = fun():
    v = if case true: 42 else: "PASS" end
    -- `v` has the shape: `Int | Str`
end

```

> [!TODO]
> We need to implement sum shapes.

## Functions

> [!TODO]
> Write what's needed to know about functions.

### Anonymous Functions

Because anonymous functions are so common, Klarer has a special syntax for them.

```todo

call_me = fun(value {}, f fun(value {})):
    f(value)
end

main = fun():
    call_me("PASS", => print($0))
end

```

```
PASS
```

An anonymous function can be passed to function calls only. You cannot assign it to a variable.
(You would just use the regular function syntax: `my_func = fun(<parameters>) <result>: ... end`)

Arguments to anonymous functions are _not_ named but positional (`$0, $1, ...`). If you
feel the need to name them, use a named function.

Anonymous functions need no `end` keyword. They naturally end at the end of the call argument
they’re part of — either the next comma or closing parenthesis.

```todo

call_me = fun(value {}, f fun(p1 {}, p2 Int), value2 Int):
    f(value, value2)
end

main = fun():
    -- A multi-expression anonymous function in the middle of a call.
    call_me(
        "PASS",
        =>
            print($0)
            print($1),
        42
    )
end

```

```
PASS
42
```

## Immutability

Klarer does not have _mutable_ values. But don't worry, it will not feel too awkward. :)

```todo

main = fun():
    -- Every assignment creates a new binding.
    person = 42
    person = {name = "John", age = 42, address = {street = "Main", city = "London"}}

    -- Shorthand syntax for replacing fields.
    -- This creates a copy of `person` and replaces the `name` field.
    person = person | {name = "Jane"}

    -- Or even shorter:
    person |= {name = "Jane"}

    -- This works with nested shapes.
    person |= {address.street = "Second"}

    -- Or in a longer form:
    person |= {address |= {street = "Second"}}

    -- Even longer:
    person = person | {address = person.address | {street = "Second"}}
end

```

### Scoping

Every assignment creates a new binding _in the current scope_.

```klarer

main = fun():
    -- This creates a new binding.
    name = "John"

    if case true:
        -- This creates a new binding in the inner scope.
        -- This _does not_ somehow update `name` in the outer scope.
        name = "Jane"
        print(name)
    end

    -- Name is still "John" here.
    print(name)
end

```

```
Jane
John
```

### Mutation Blocks

Though powerful, the "copy and replace" syntax (`|` and `|=`) is sometimes not enough. But
rebuilding whole object trees because some object in the middle of the tree changed is neither
efficient nor elegant.

Klarer introduces the concept of _mutation blocks_ to solve this problem. Think of mutation blocks
as _localized mutability_.

```todo

-- Update all numbers in place.
increment_all = fun(nums [Int]) [Int]:
    result = mutate nums:
        for i in 0..nums.len():
            nums[i] = nums[i] + 1
        end
        nums
    end
    result
end

main = fun():
    nums = [1, 2, 3]
    print(nums.join(", "))
    increment_all(nums)
    print(nums.join(", "))
end

```

```
1, 2, 3
2, 3, 4
```

A more complex example:

```todo

Tree = {id Int, left Tree|:none, right Tree|:none, count Int}

-- This is how you would increment the count of a deeply nested tree node
-- in a functional way. You basically rebuild the tree.
inc_count_functional = fun(tree Tree, id Int) Tree:
    if case tree.id == id:
        tree |= {count = tree.count + 1}
    else:
        tree |= {
            left = match tree.left:
                case {:none}: :none
                case left: inc_count_functional(left, id)
            end,
            right = match tree.right:
                case {:none}: :none
                case right: inc_count_functional(right, id)
            end
        }
    end
end

inc_count_mutate_block = fun(tree Tree, id Int) Tree:
    mutate tree:
        find_and_inc = fun(node Tree | :none):
            match node:
                case {:none}: return
                case n:
                    if case n.id == id:
                        n.count = n.count + 1  -- Direct mutation!
                    else:
                        find_and_inc(n.left)
                        find_and_inc(n.right)
                    end
            end
        end

        find_and_inc(tree)
        -- Return the mutated tree.
        tree
    end
end

main = fun():
    -- Build a tree:      1
    --                   / \
    --                  2   3
    tree = Tree{
        id = 1,
        count = 0,
        left = Tree{id = 2, count = 0, left = :none, right = :none},
        right = Tree{id = 3, count = 0, left = :none, right = :none}
    }

    -- Traditional: rebuilds the tree.
    tree1 = inc_count_functional(tree, 2)
    print(tree1.left.count)

    -- Mutation block: modifies in place, returns modified tree.
    tree2 = inc_count_mutate_block(tree, 2)
    print(tree2.left.count)
end

```

```
1
1
```

#### Rules for Mutation Blocks

A mutation block is a block of code that can mutate a value that is passed to it.

Semantically, that value is _copied_ before the mutation block is executed and the
result of the mutation block is also a _copy_. The compiler will most likely _not_
make copies but the guarantee is that the mutated value does not escape the mutation block.

Inside a mutation block, you cannot:

- Call functions that are defined outside the mutation block.
- Can't leak the value that is passed to the mutation block, i.e. you cannot assign
  it to anything that is returned from the mutation block.

#### How Do Mutation Blocks Work?

Inside a mutation block, the normal immutability rules are suspended _for the input value_.
You can directly mutate fields using =, and these mutations affect the value being worked on.
When the block completes, it returns the mutated value as if it were newly created.

#### Use Cases for Mutation Blocks

Use mutation blocks where updating a (deeply nested) field results in a complex rebuild of the
input value. Basically, if the code is easier to write and understand with mutation blocks, then
it is a good idea to use them.

Another use case are algorithms that are either not tail recursive or are straight up faster.

A good example is the quicksort algorithm. Though it reads quite elegantly in a functional style,
it benefits from mutation blocks because you can swap values in place. (The compiler _might_
optimize the functional version to use in-place swaps, but that is not guaranteed and sometimes not
even possible.)

```todo

-- Traditional immutable quicksort - quite elegant but potentially slow.
quicksort_functional = fun(arr []Int) []Int:
    if case arr.len() <= 1:
        arr
    else:
        pivot = arr[0]

        less = arr.filter(=> $0 < pivot)
        equal = arr.filter(=> $0 == pivot)
        greater = arr.filter(=> $0 > pivot)

        quicksort_functional(less) + equal + quicksort_functional(greater)
    end
end

-- Mutation block version - in-place sorting.
quicksort_mutate = fun(arr []Int) []Int:
    mutate arr:
        sort = fun(low Int, high Int):
            if case low >= high: return end

            -- Partition around pivot.
            pivot = arr[high]
            i = low - 1

            j = low
            loop:
                if case j >= high: break end
                if case arr[j] <= pivot:
                    i = i + 1

                    -- Swap arr[i] and arr[j].
                    temp = arr[i]
                    arr[i] = arr[j]
                    arr[j] = temp
                end
                j = j + 1
            end

            -- Place pivot in final position.
            i = i + 1
            temp = arr[i]
            arr[i] = arr[high]
            arr[high] = temp

            -- Recursively sort partitions.
            sort(low, i - 1)
            sort(i + 1, high)
        end

        sort(0, arr.len() - 1)
        arr
    end
end

main = fun():
    data = [3, 1, 4, 1, 5, 9, 2, 6]

    sorted1 = quicksort_functional(data)
    print(sorted1.join(", "))

    sorted2 = quicksort_mutate(data)
    print(sorted2.join(", "))
end
```

```
1, 1, 2, 3, 4, 5, 6, 9
1, 1, 2, 3, 4, 5, 6, 9
```
