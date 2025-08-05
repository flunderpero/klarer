# Klarer

Klarer aims to be a beautiful, fast, and all-purpose programming language.

## Goals

Klarer should be:

- Easy to read and write
- Fast to compile
- Fast to execute (around the speed of Go)

# Shapes and Behaviours

Klarer programs are built from two fundamental concepts: **Shapes** and **Behaviours**.

## Shapes

A Shape in Klarer is a named structural definition of attributes. It is pure data. A Shape defines the structure of a value, such as a List or a Tree, but it does not carry any behavior or methods by itself.

```klarer
List = {elements []{}, index Int}
```

This defines a `List` shape with an array of elements and an index. This is just a structure; it does not inherently know how to push or pop elements. Shapes are only data.

## Type Inference Through Usage

When a value is created, Klarer infers its shape from the attributes given at initialization. Attributes that are structures (like arrays or nested shapes) start as `{}` (an unconstrained shape) until refined through actual usage.

Function parameters do not declare types explicitly. The compiler infers parameter shapes by analyzing how they are used inside the function body. Field access, function calls, and assignments narrow the shape.

Example:

```klarer
List.push = (list, e) => list.elements[list.index] = e list.index += 1;
```

The shape of `list` is inferred because `.elements` and `.index` are accessed. The shape of `.elements` is refined based on the first assignment to `e`. Any incompatible shape assignments will cause a compile-time error unless disambiguated by atoms.

There are no generics. Klarer relies entirely on structural refinement through usage.

## Behaviours

Behaviours are function namespaces that can be attached to Shapes or values. They are written with an `@` prefix, like `@List` or `@Tree`.

Behaviours are not types or interfaces. They are libraries of functions that declare structural expectations implicitly through how they use their parameters.

```klarer
@List.push = (list, e) => ...
@List.pop = (list) => ...
```

Behaviours do not automatically attach to any shape. They must be explicitly attached.

## Explicit Behaviour Attachment

To use functions from a Behaviour, the Behaviour must be attached to a Shape alias or an individual value.

Attach at alias level:

```klarer
List = {elements []{}, index Int} + @List
```

Attach at value level:

```klarer
mylist = {elements []{}, index 0} + @List
```

Behaviours never attach implicitly. The developer must always declare intent.

## Structural Method Availability

Attaching a Behaviour does not make all its functions available. Each function is only available if the shape (or value) structurally satisfies the requirements of that function.

```klarer
@Tree.debug = (tree) => tree.left.to_str() + tree.right.to_str()
```

If a shape does not provide `.to_str()` on `.left` and `.right`, the `.debug()` method will not be available. The compiler provides a clear error if a method is called that cannot be attached due to missing structural requirements.

## Dot-Syntax as Syntactic Sugar

Dot-syntax (e.g., `x.method()`) is syntactic sugar for calling a function from an attached Behaviour namespace.

`x.method(args...)` resolves to `@Behaviour.method(x, args...)` if the Behaviour is attached and the shape satisfies the function's requirements.

Dot-syntax does not imply object-orientation. It is purely a syntactic convenience. There is no runtime dispatch.

## Behaviour Resolution Order

When multiple Behaviours are attached, Klarer resolves method calls by searching Behaviours in the order they are attached (left to right).

```klarer
mylist = List{elements []Str, index 0} + @List + @Iterator
```

Here, `.join_str()` will resolve to `@List.join_str()` if available. If not, Klarer will try `@Iterator.join_str()`. The first match wins.

There is no specialization ranking. The developer controls precedence explicitly by the order of attachment.

## Summary

Shapes are data. Behaviours are function namespaces. Attaching a Behaviour to a shape is a deliberate act. Method availability is governed by structural compatibility, not by type declarations. Dot-syntax is syntactic sugar over behaviour function calls. Behaviours are resolved in the order they are attached, ensuring predictability and clarity.
