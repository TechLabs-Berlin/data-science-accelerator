# Handling input and testing your code

As Data Scientist you will write a lot of code, that you will use. You will also write code that others may use or which will receive input from a database or user.

How can you verify that your code is correct? How can you make sure that the input your functions receive are correct?

One way is to set variable types, another way is automatic testing. We will focus on the latter.

There are elaborate testing frameworks, but for now we will show you a simple but effective testing method: 

**`assert` statements**

These statements test whether a certain condition is true and will throw an `AssertionError` if this is not the case.

`assert 1 + 3 == 4` -> No error
`assert 1 + 3 == 5` -> `AssertionError`  (and an angry letter to your first-grade teacher)

You can add an optional error message to your statements. This error message will be displayed if an `AssertionError` is raised

`assert 1 + 3 == 4`, " 1 + 3 should equal 4, but it didn't"

Now verifying first grade math is not very exiting, but you can extend this logic to more complicated things.
