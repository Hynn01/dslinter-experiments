#!/usr/bin/env python
# coding: utf-8

# The single best thing you can do to make your code more professional is to make it **reusable**.
# 
# What does “reusable” mean? At some point in your data science career, you’re going to write code that will be used more than just once or twice. Maybe you’re running the same preprocessing pipeline on some different sets of image files, or you’ve got a suite of evaluation techniques that you use to compare models. We’ve all copied and pasted the same code, but **once you find yourself copying the same code more than once or twice, it’s time to sink some time into making your code reusable**. Reusing well-written code isn’t cheating or slacking: it’s an efficient use of your time and [it’s considered a best practice](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself) in software engineering.
# 
# There are six central principles that I think 1) make it easy for you or your colleagues to reuse your code, 2) make your code look really polished and professional and, above all, 3) **save you time**.
# 
#   * 📦 **Modular**: Code is broken into small, independent parts (like functions) that each do one thing. Code you’re reusing lives in a single central place.
#   * ✔️ **Correct**: Your code does what you say/think it does.
#   * 📖 **Readable**: It’s easy to read the code and understand what it does. Variable names are informative and code has up-to-date comments and [docstrings](https://www.python.org/dev/peps/pep-0257/).
#   * 💅 **Stylish**: Code follows a single, consistent style (e.g. the [Tidyverse style guide](https://style.tidyverse.org/) for R, [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code)
#   * 🛠️ **Versatile**: Solves a problem that will happen more than once and anticipates variation in the data.
#   * 💡 **Creative**: Solves a problem that hasn’t already been solved or is a clear improvement over an existing solution.
# 
# Let’s go through each of these steps in a bit more detail.with a bit of sample code and see how they work in practice.
# 

# ## 📦 Modular
# 
# Modular code means that your code is broken into small, independent parts (like functions) that each do one thing. 
# 
# Each function, whether in Python or R, has several parts:
# 
# * A *name* for the function.
# * *Arguments* for your function. This is the information you’ll pass into your function. 
# * The *body* of your function. This is where you define what your function does. Generally, I’ll write the code for my function and test with an existing data structure first and then put the code into a function. 
# * A *return value*. This is what your function will send back after it’s finished writing. In Python, you’ll need to specify what you want to return by adding `return(thing_to_return)` at the bottom of your function. In R, by default the output of the last line of your function body will be returned. 
# 
# Let’s look at some examples. Here are two sample functions, one in Python & one in R, that do the same thing (more or less).
# 
# * They both have the same function name, `find_most_common`
# * They both have one argument, `values`
# * They both have a body that does roughly the same thing: count how many times each value in `values` shows up
# * They both return the same thing: the value(s) that is most common in the input argument `values`
# 
# ### Python Example
# 
# ```
# # define a function
# def find_most_common(values):
#     list_counts = collections.Counter(values)
#     most_common_values = list_counts.most_common(1)
#     
#     return(most_common_values[0][0])
# 
# # use the function
# find_most_common([1, 2, 2, 3])
# ```
# 
# ### R Example
# 
# ```
# # define the function
# find_most_common <- function(values){
#   freq_table <- table(values)
#   most_common_values <- freq_table[which.max(freq_table)]
# 
#   names(most_common_values)
# }
# 
# # use the function
# find_most _common(c(1, 2, 2, 3))
# ```
# Pretty straightforward, right? (Even if the syntax between the two languages is a little different). You can use this general principle of writing little functions that do one thing each to break your code up into smaller pieces. 
# 
# ### Why functions? 
# 
# If you have some more programming experience, you may be curious why I choose to talk about functions instead of classes or other related concepts from [object oriented programming]. I think functional programming tends to be a very natural fit for a lot of data science work so that’s the general framework I’m going to use to show you examples of modular code.
# 
# > **Functional programming.**  \A style of writing code where you pass one or more pieces of data into a function and the result you get back will be some sort of transformation of those pieces of data. This means that you wouldn’t do things like modifying an existing variable in the body of a function. If you’re interested in learning more, I’d recommend [this talk on functional programming for data science](https://www.youtube.com/watch?v=bzUmK0Y07ck). 
# 
# The main reason that I like using a functional approach for data science is that it makes it easy to start chaining together multiple functions into a data processing pipeline: the output of one function becomes the input to the next. Something like this:
# 
# > data -> function 1 -> function 2 -> function 3 -> transformed data
# 
# There are some very helpful tools to help you do this, including [pipes in R](https://magrittr.tidyverse.org) and [method chaining from the pyjanitor in Python](https://pyjanitor.readthedocs.io/notebooks/pyjanitor_intro.html#Clean-up-our-data-using-a-pyjanitor-method-chaining-pipeline).
# 
# ### Python example: chaining functions together
# 
# This example is based on one from [the pyjanitor documentation](https://pyjanitor.readthedocs.io/notebooks/pyjanitor_intro.html#Clean-up-our-data-using-a-pyjanitor-method-chaining-pipeline) and shows you how to set up a little data pipeline using existing Pandas functions.
# 
# It reads in a file (the `pd.read_excel('dirty_data.xlsx')` line) and then transforms it using a number of functions that clean the column names, remove missing data, renames one of the columns and converts one of the columns to datetime format. The output is also a dataframe.
# 
# ```
# cleaned_df = (
#     pd.read_excel('dirty_data.xlsx')
#     .clean_names()
#     .remove_empty()
#     .rename_column("full_time_", "full_time")
#     .convert_excel_date("hire_date")
# )
# 
# cleaned_df
# ```
# ### R example: chaining functions together
# 
# And here’s an R example that does the same thing as the Python example. 
# 
# ```
# cleaned_df <- read_excel('dirty_data.xlsx') %>%
#   clean_names() %>%
#   remove_empty() %>%
#   renames(“full_time”, “full_time_”) %>%
#   excel_numeric_to_date(“hire_date”)
# ```
# 
# > Breaking your code apart into functions--particularly if each function just transforms the data that gets passed into it--can save you time by letting you reuse code and combine different functions into compact data pipelines,
# 

# ## ✔️ Correct
# 
# By “correct” I mean that your code does what you say/think it does. This can be tricky to check. One way to make sure your code is correct is through [code review](https://medium.com/apteo/code-reviewing-data-science-work-774747248e33).
# 
# > **Code review** is a process where one of your colleagues carefully checks over your code to make sure that it works the way you think it does.
# 
# Unfortunately, that’s not always practical for data scientists. Especially if you’re the only data scientists in a company, it would be tough to get someone without a statistics or machine learning background to the point where they could give you expert feedback on your code. As the field grows larger it may become more common for data science code to undergo code review… but the meantime you can help make sure your code is correct by including some tests.
# 
# > **Tests** are little pieces of code you use check that your code is working correctly 
# 
# Writing tests doesn't have to be complex! Here, I’m going to work through how to add a test to a function with just a single line of code.
# 
# (The example I’m going to work on here is in Python, if you’re looking for an R example, or even just more discussion of where testing fits in the data science workflow, [check out this vignette by Hadley Wickham](http://r-pkgs.had.co.nz/tests.html).)
# 
# In the Python function I wrote above, I returned the most common value… but what if there was more than one value tied for the most common? Currently our function will just return one of them, but if I really need to know if there’s a tie my current function won’t do that. 
# 
# So let’s include a test to let us know if there’s a tie! `assert` is a method [built into Python](https://docs.python.org/3/reference/simple_stmts.html#the-assert-statement) that will let us check that something is true. If it is, nothing happens. If it isn’t, our function will stop and give us an error message.

# In[ ]:


import collections

def find_most_common(values):
    """"Return the value of the most common item in a list"""
    list_counts = collections.Counter(values)
    top_two_values = list_counts.most_common(2)

    # make sure we don't have a tie for most common
    assert top_two_values[0][1] != top_two_values[1][1]        ,"There's a tie for most common value"
    
    return(top_two_values[0][0])


# The `assert` statement here is checking that the count of the most common value isn’t the same as the count of the second most common value. If it is, the function stops and returns an error message.
# 
# First, let’s check that our function will work as expected if there’s not a tie: 
# 

# In[ ]:


values_list = [1, 2, 3, 4, 5, 5, 5]

find_most_common(values_list)


# So far so good: there are more 5's than any other values. But what if there's a tie?

# In[ ]:


values_list = [1, 2, 3, 4, 4, 4, 5, 5, 5]

find_most_common(values_list)


# We get an assertion error and the nice, helpful error message we wrote for ourselves earlier!
# 
# While this is a pretty simple example, including some tests can help you make sure that your code is doing what you think it’s doing. This is particularly important if you’re importing other libraries and updating them regularly: just because you didn’t change your code it doesn't mean that the code you’re importing hasn’t changed! Tests can help you find bugs before they end up creating problems. 
# 
# > Using tests to check that your code is correct can help you save time by catching bugs quickly.

# ## 📖 Readable
# 
# “Readable” code is code that is easy to read, even if it’s the first time you’ve seen it. In general, the more things like variable and function names are words that describe what they do/are the easier it is to read the code. In addition, comments that describe what the code does at a high level or why you made specific choices can help you 
# 
# ### Variable names
# 
# Variable names are informative and code has up-to-date comments and [docstrings](https://www.python.org/dev/peps/pep-0257/).
# 
# Some examples of not-very-readable variable names are:
# 
# * **Single characters**, like `x` or `q`.  [There are a few exceptions](https://www.codereadability.com/i-n-k-single-letter-variable-names/), like using `i` for index or `x` for the x axis.
# * **All lower case names with no spaces between words** `likethisexample` or `somedatafromsomewhere`
# * **Uninformative or ambiguous names** `data2` doesn’t tell you what’s in the data or how it’s different from `data1`. `df` tells you that something’s a dataframe… but if you have multiple dataframes how can you tell which one?
# 
# You can improve names by following a couple of rules:
# 
# * Use some way to **indicate the spaces between words** in variable names. Since you can’t use actual spaces, some common ways to do this are `snake_case` and `camelCase`. Your style guide will probably recommend one. 
# * Use the names to **describe what’s in the variable or what a function does**. For example, `sales_data_jan` is more informative than just `data`, and `z_score_calculator` is more informative than just `calc` or `norm`. 
# 
# It’s ok to have not-ideal variable names when you’re still figuring out how you’re going to write a bit of code, but I’d recommend going back and making the names better once you’ve got it working.
# 
# ### Comments
# 
# Comments are blocks of natural language text in your code. In both Python and R, you can indicate that a line is a comment by starting it with a #. Some tips for writing better comments:
# 
# * While some style guides recommend not including information on what a bit of code is doing, I actually think that it’s often warranted in data science. I personally **include comments describing *what* my code is doing if**:
#   * I’m using a relatively new method, especially for modelling. It can also be helpful to include a link to reference material for the method. (I look back to the papers/blog posts I linked all the time.)
#   * My colleagues who are working with me on a project aren’t familiar with the programming language I’m using. (Not everyone knows Python or R!)
# * **If you change the code, remember to update the comment!**
# * If you’re using an uncommon way of doing something it’s worth adding a comment to explain why so someone (which could be you!) doesn’t run into problems later if they try to update the code. For example: `# using tf 1, don’t update to 2 until bug #1234 is closed or it will break the rest of the pipeline`
# * Some style guides will recommend only ever writing comments in English, but if you’re working with folks who use another language I’d personally suggest that you **write comments in whichever language will be easiest for everyone using the code to understand**.
# 
#  **Docstring:** In Python, a docstring is a comment that’s the first bit of text in a function or class. If you’re importing functions, you should include a docstring. This lets you, and anyone else using the function, quickly learn more about what the function does.
# 
# ```
# def function(argument):
# 	“”” This is the docstring. It should describe what the function will do when run”””
# ```
# To check the docstring for a function, you can use the syntax `function_name.__doc__`. 
# 
# If you’re an R user and what to add docstrings to your code, you can check out [the docstring package](https://cran.r-project.org/web/packages/docstring/vignettes/docstring_intro.html). 
# 
# > Readable code is faster to read. This saves you time when you need to go back to a project or when you’re encountering new code for the first time and need to understand what’s going on.
# 

# 
# ## 💅 Stylish
# 
# When I say “stylish” here I literally mean “following a specific style”. Styles are described and defined in documents called “style guides”. If you haven’t used a style guide before, they’re very handy! Following a specific style guide makes your code easier to read and helps you avoid common mistakes. (It can also help you avoid writing code with [code smells](https://en.wikipedia.org/wiki/Code_smell).)
# 
# Style guides will offer guidance on things like where to put white spaces, how to organize the structure of code within a file and how to name things like functions and files. Code that doesn’t consistently follow a style guide may still run perfectly fine, but it will look a little odd and generally be hard to read.
# 
# > Pro tip: You can actually use a program called a “linter” to automatically check if your code follows a specific style guide. Pylint for Python & lintr for R are two popuilar linters. You can see [an example of how to use a linter to check an R utility script here](https://www.kaggle.com/rtatman/linting-scripts-in-r).
# 
# Once you’ve picked a style guide to follow, you should do your best to follow it consistently within your code. There are, of course, differences across style guides, but a couple things do tend to be the same across both Python and R style guides. A couple examples:
# 
# * You should have all of your imports (`library(package_name)` or `import module_name`) at the top of your code file and only have one import per line.
# * Whether you indent with tabs or spaces will depend on your style guide, but you should never mix tabs and spaces (e.g. have some lines indented with two spaces and some lines indented with a tab).
# * Avoid having spaces at the ends of your lines
# * Function and variables names should all be lower case and have words seperated_by_underscores (unless you’re working with existing code that follows another standard, in which case use that)
# * Try to keep your lines of code fairly short, ideally less than 80 characters long
# 
# Style guides can be a little overwhelming at first, but don’t stress too much. As you read and write more code it will become easier and easier to follow a specific style guide. In the meantime, **even a few small improvements will make your code easier to follow and use**. 
# 
# ### Example
# 
# For this example, we’re going to be using some R code and modifying it to fit the [Tidyverse style guide](https://style.tidyverse.org).
# 
# ```
# CZS <- function(x) {
#    sd <- sd(x); m = mean(x)
# (x -m)/sd}
# ```
# 
# There are quite a few things we can fix here so that they follow the Tidyverse style guide. 
# 
# * The function name isn’t informative and doesn’t follow the Tidyverse conventions (lower_case_with_underscores).
# * We’re using multiple assignment operators (<- and =).
# * We’re using a mix of tabs and spaces.
# * We’ve concatenated multiple lines using `;` (this is possible but *strongly discouraged* in both Python and R).
# * We don’t have spaces around all our infix operators (mathematical symbols like `+`, `-`, `\`, etc.).
# * We don’t have the closing curly brace, `}`, on a new line.
# 
# Once we’ve tidied these things up, our code now looks like this:
# 
# ```
# calculate_z_score <- function(x) {
#     sd <- sd(x)
#     m <- mean(x)
# 
#     (x - m) / sd
# }
# ```
# I personally find this a lot easier to read than the first example, even though they do the exact same thing.  
# 
# > Stylish code is generally faster to read and find bugs in. In addition, most style guides recommend best practices to help you avoid common bugs. All of this saves you and your colleagues time in debugging.
# 

# ## 🛠️ Versatile
# 
# “Versatile” means useful in a variety of situations. Versatile code solves a problem that will happen more than once and anticipates variation in the data. 
# 
# ### Should I only ever write code if I’m planning to reuse it?
# 
# No, of course not. There’s nothing wrong with writing new code to solve a unique problem. Maybe you need to rename a batch of files quickly or someone’s asked you to make a new, unique visualization for a one-off presentation. 
# 
# However, you probably don’t want to go to all the trouble of making every single line of code you ever write totally polished and reusable. [While some folks would disagree with me](https://www.youtube.com/watch?v=Sg6xJ0ACc78) **I personally think it’s only worth spending a lot of time polishing code if you--or someone else--is actually going to reuse it**. 
# 
# Data scientists have to do and know about a lot of different things: you’ve probably got a better use for your time than carefully polishing every line of code you ever write. Investing time in polishing your code starts to make sense when you know the code is going to be reused. A little time spent making everything easier to follow and use while it’s still fresh in your head can save a lot of time down the line.
# 
# ### Anticipating variation in the data
# 
# By “variation in the data” I mean differences in the data that will break things down the line. For example, maybe you’ve written a function that assumes that your dataframe has a column named `latitude`. If someone changes the name of the column to `lat` in the database next week, your code may break. 
# 
# In order to make sure that you’re getting the data in you expect to be getting in, you can use data validation. I have [a notebook here](https://www.kaggle.com/rtatman/automating-data-pipelines-day-2#Scripting-your-data-validation) that covers data validation in more detail if you’re curious, but here are a few of my favorite tools.
# 
# ---
# 
# **Python:**
# 
# * I like the csvvalidator package, which [I’ve previously written an introduction to](https://www.kaggle.com/rtatman/dashboarding-with-notebooks-day-5). 
# * For JSON data in Python, the [Cerberus module](http://docs.python-cerberus.org/en/stable/usage.html) is probably the most popular tool. 
# * For visualizing missing data in particular, [the missingno package](https://github.com/ResidentMario/missingno) can be very handy. 
# * To check the type of your file the [python-magic module](https://github.com/ahupp/python-magic) can be helpful.
# 
# **R:**
# 
# * For R, [the validate package](https://cran.r-project.org/web/packages/validate/vignettes/introduction.html) for data validation ([which I’ve previously written a tutorial for](https://www.kaggle.com/rtatman/dashboarding-with-notebooks-day-5-r)) is probably your best bet. It can handle tabular, hierarchical and just raw text data, which is nice. :)
# * To figure out the file type, [guess_types from the mime package](https://www.rforge.net/doc/packages/mime/guess_type.html) can be helpful.
# 
# --- 
# 
# > Versatile code can be used in a variety of situations. This saves you time because you can apply the same code in multiple different places. 
# 

# ## 💡 Creative
# 
# By “creative”, I mean code that solves a problem that hasn’t already been solved or is a clear improvement over an existing solution. The reason that I include this is to encourage you to seek out existing libraries or modules (or [Kaggle scripts](https://www.kaggle.com/kernels?sortBy=hotness&group=everyone&pageSize=20&tagIds=16074)!) that already exist to solve your problem. If someone has already written the code you need, and it’s under a license that allows you to use it, then you should probably just do that. 
# 
# I would only suggest writing a library that replicates the functionality of another one if you’re making a clear improvement. For example, the [Python library flashtext](https://flashtext.readthedocs.io/en/latest/). It allows you to do the same thing as you can with [regular expressions](https://en.wikipedia.org/wiki/Regular_expression)--like find, extract and replace text--but [much, much faster](https://github.com/vi3k6i5/flashtext#why-not-regex). 
# 
# > Only spending time writing code if there’s no existing solution saves you time because you can build on existing work rather than starting over from scratch.

# 
