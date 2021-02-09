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

`assert 1 + 3 == 4, "1 + 3 should equal 4, but it didn't"`

Now verifying first grade math is not very exiting, but you can extend this logic to more complicated things.

Let's imagine we have a function that computes the cosine similarity between the last couple columns in two dataframes. These columns hold our numeric features, which we want to compare for similarity.

We could start our function with something like this `def create_sim_score(df1,df2,scoring = "cosine_sim"):`

Now we can use an assert statement with an error message to make sure that the two dataframes have the same columns (by name in this case):
`assert list(df1.columns[6:]) == list(df2.columns[6:]), "feature columns in both dataframes need to be equal"`

This still is not a very involved example but it shows how you can check the input of a function quickly and give a **meaningful** error message in case something goes wrong.

Try to think of ways you can incorporate little checks and meaningful error messages in your work!
They can be really helpful for others but also for your future self looking at old code.
