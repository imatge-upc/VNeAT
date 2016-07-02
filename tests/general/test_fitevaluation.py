from src.FitScores.FitEvaluation_v2 import evaluation_function as evaf


@evaf
def my_eval(self):
    """This is the my_eval function.
    """
    return self.three() + 5


my_eval.requires('three', 'A method that returns three')


class A(object):
    pass


a = A()

# default value for any class (inherited from object) and any evaluation function (checked in evaf)
evaf[object].bind('three', lambda _: 3)
# Notice that the bound method takes one argument: in general, any bound methods will receive th
# bound evaluation method itself (in this case, evaf[object]) as first argument, and any other arguments
# specified by the 'requires' clause (in the description) afterwards. It is the user's task to ensure
# the correctness of the arguments (in this case, the evaluation method is ignored; scroll down -almost until
# the end of the script- to see a more complete example)

print evaf[object].status()
print my_eval[a].status()

# Both should inherit from object and therefore output 8 (3 + 5)
print my_eval[a].evaluate()
print my_eval[A].evaluate()

# OK, let's make it interesting
my_eval[object].bind('three', lambda _: -3)

# Those should print 2 (-3 + 5)
print my_eval[a].evaluate()
print my_eval[A].evaluate()

# What about this?
evaf[A].bind('three', lambda _: 5)

# So we specified that for any object, when evaluating my_eval, three should return -3, whereas
# for any test, when evaluating the class A (and any object that inherits from it), three should output 5

# In this case, the last condition takes preference, since it is hypothesized that whoever bound
# a particular method to a given requirement for a specific object is more specialized in that particular
# object and therefore knows better what method is appropriate for that requirement (at least they know
# it better than whoever bound the other method in a much more generic way to the test itself "for all objects")

# Therefore, both of these should output 10 (5 + 5)
print my_eval[a].evaluate()
print my_eval[A].evaluate()

# Now, if we set 'three' for an instance
evaf[a].bind('three', lambda _: 1)

# That should not affect the corresponding class
print my_eval[a].evaluate()  # 1 + 5 = 6
print my_eval[A].evaluate()  # 5 + 5 = 10 (from previous data)

# By the way, setting a generic value for a (potential) requirement through evaf will make the status show the corresponding
# requirement as Inherited rather than Bound ('three' should appear as [Inherited] in both cases):
print my_eval[a].status()
print my_eval[A].status()

# The only case in which this appears as Bound is when we specify the requirement for the current test and class/object
# explicitly. Let's check it out!

my_eval[A].bind('three', lambda _: 95)
print my_eval[A].status()  # 'three' should appear as [Bound]

# Notice that, again, the generic method on a specific object takes precedence over the specific method on
# a more generic object (such as the class 'A' as opposed to its instance 'a')

print my_eval[A].evaluate()  # 100 (95 + 5)
print my_eval[a].evaluate()  # 6 (1 + 5)

# Now, let's try something different

my_eval[A].clear()
print my_eval[A].status()  # 'three' is now inherited from its generic version (evaf[A])
print my_eval[A].evaluate()  # as a result, this should output 10 again (5 + 5)

evaf[A].unbind('three')  # now A should inherit from the {my_eval<->object} binding
print my_eval[A].evaluate()  # -3 + 5 = 2

evaf.clear()  # After this there are not generic bindings at all
print my_eval[A].status()  # But A still inherits from the {my_eval<->object} binding
print my_eval[A].evaluate()  # 2 (same as previous evaluation)

# OK, let's delete that last chain-link
my_eval[object].clear()
print my_eval[A].status()  # Now it should appear as [Required] (i.e., required and unbound)
try:
    print my_eval[
        A].evaluate()  # And this should raise an error since 'three' cannot be inherited and is not bound to A for my_eval
except RuntimeError as e:
    print 'RuntimeError raised:', e


# Well, let's change the scenario now

# We first define a new evaluation function
@evaf
def my_new_eval(self):
    try:
        return self.three() + 100 * self.pi()
    except AttributeError:
        return 101 * self.pi()  # pi ~ 3.1416 (we approximate 3 by using pi, weirdly enough :D)


# We set the method pi to return 3.1416 by default
my_new_eval.implicit('pi', 'A method that returns the mathematical constant PI.', lambda _: 3.1416)

# BUT we do NOT require the method 'three'; this way, if an object defines it, we can use it, and otherwise we can
# use the method 'pi' to estimate the value of 3 approximately (this is weird, as we said, but it's just an example).



my_new_eval[object].unbind('three')  # raises error, 'three' method was not bound to object for my_new_eval
my_new_eval[object].bind('three', lambda _: 3)  # raises error, 'three' method is not required by object for my_new_eval
my_new_eval[object].bind('three', lambda _: 3, force=True)  # but we can force the binding
print my_new_eval[object].status()  # this will show as [Forced]

# And we can even evaluate methods that did not specify a given requirement with forced bindings (consider this as optional bindings)
print my_new_eval[object].evaluate()  # should output 100*pi + 3 ~ 317.16

# However, inheritance won't work in this case:
print my_new_eval[A].evaluate()  # outputs 317.3016 (~ 101*pi), since 'three' cannot be found

# We can also override the default value for a specific object
my_new_eval[A].bind('pi', lambda _: 3.141592654)
print my_new_eval[A].evaluate()  # outputs 317.300858054 (closer to the real value of 101*pi than before)

# Nevertheless, in this case inheritance won't work either (implicits take relevance over inheritance)
print my_new_eval[a].evaluate()  # outputs 317.3016

# Moreover, we are able to access the elements of the bound evaluation function itself, including the target
# and its attributes (through the self.target variable), from the bound method:
my_new_eval[a].bind('three', lambda self: self.target.three - 100 * self.pi(), force=True)
print my_new_eval[a].status()
print my_new_eval[a].evaluate()  # outputs 317.3016
# why? because 'three' has not been defined in target (object 'a')
# therefore, evaluating self.target.three (a.three) raises an AttributeError, which is captured in the evaluation
# function (my_new_eval), that returns 101*self.pi() instead of the original expression

# If we set a.three,
a.three = 3
print my_new_eval[a].evaluate()  # outputs 3.0
# the computed expression becomes self.three() + 100*self.pi() = self.target.three - 100*self.pi() + 100*self.pi()
# i.e., the result is self.target.three = a.three = 3



# OK, so here's a new thing:
# When we are evaluating a specific test over a given object/class, we may want to do so based on some results
# To this end, we can also pass an arbitrary object as the 'fitting_results' argument to the 'evaluate' method
# Such object is accessible as an attribute (called the same) of the binding; here's an example to show this:

my_new_eval[a].bind('three', lambda self: self.fitting_results.result_three - 100 * self.pi(), force=True)
# If we call it without an argument, this will raise an AttributeError, since the 'fitting_results' attribute won't be set;
# As before, the AttributeError will be captured in the my_new_eval method, and 101*pi will be printed
print my_new_eval[a].evaluate()  # outputs 317.3016


# However, when calling it with an object that does have the 'result_three' attribute, this will work
class R:
    pass


r = R()
r.result_three = 3

print my_new_eval[a].evaluate(r)  # outputs 3.0


class C1(object):
    pass


class C2(C1):
    pass


@evaf
def eval1(self):
    return self.f()


eval1.requires('f', 'Just any function')
eval1.requires('x', 'Just another function')
eval1.requires('y', 'Just another another function')

eval1[C1].bind('f', lambda self: self.x()).bind('x', lambda self: 3, force=True).bind('y', lambda self: 10)

eval1[C2].bind('x', lambda self: self.y(), force=True)

print eval1[
    C2].evaluate()  # prints 10, inheritance works correctly (although f was inherited from C1, x is evaluated in C2)
