# Taking 2 integer values to perform operations on them
val1 = int(input("Please enter 1st number: "))
val2 = int(input("Please enter 2nd number: "))

#Taking input for what operation we want to perform
var = input("""Press 1 for Addition: 
Press 2 for Substraction:
Press 3 for Multipilication:
Press 4 for Division:
Press 5 for Reminder:""")

#defining funtions to perform operations
def sum(a, b):
    return (a+b)
def sub(a, b):
    return a-b 
def mul(a, b):
    return a*b
def div(a, b):
    return a/b
def rem(a,b):
    return a%b
# Check point what operation user wants to perform and printing results accordingly

if var == "1":
    print( "Result: ", sum(val1, val2))
elif var == "2":
    print(" Result: ", sub(val1, val2))
elif var == "3":
    print(" Result: ", mul(val1, val2))
elif var == "4":
    if val2 != 0:
        print("Result:", div(val1, val2))
    else:
        print("Error: Division by zero!")
elif var == "5":
    print(" Result: ", rem(val1, val2))
else:
    print("Invalid Choice! ")
#end of program