#!/usr/bin/env python
# coding: utf-8

# # 1.Manipulate using a list

# In[1]:


ls=[2,4,6,8]
ls


# In[3]:


#1 To add new elements to the end of the list
ls.append(10)
ls.extend([12,14,16])
print("After adding, the list are",ls)


# In[11]:


#2 Reverse elements in the tuple
ls = [3,6,9]
ls.reverse()
print("after reverse the elements are",ls)


# In[15]:


#3 to display same list of elements in the list
ls= [1,3,5,6,5,3,2,1]
print("")


# In[25]:


#4 to concatenate two list
ls1= [5,1,9]
ls2= [2,3,8]
concatenated_ls = ls1+ls2 
 
print("after concatenation my two list",concatenated_ls)


# In[26]:


#5 to sort the elements in the list ascending order
ls=[9,7,5,4,3,1]
ls.sort() # Sorts the list in ascending order in-place
print("ascending",ls)


# # 4 Python program

# In[27]:


tuple1 = (10,50,20,40,30)


# In[28]:


#1 To display the elements 10 and 50 from tuple1
print(tuple1[0], tuple1[2])


# In[29]:


#2 To display the length of a tuple1
print(len(tuple1))


# In[31]:


#3 To find the minimum element from tuple1
print(min(tuple1))


# In[32]:


#4 To add all elements in the tuple1
print(sum(tuple1))


# In[34]:


#5 To display the same tuple1 multiple times
print(tuple1*3)


# #  3 Python program to implement the following using list

# In[35]:


#1  Create a list with integers
numbers = [3,1,5,3,7,9,4,2,0,6]


# In[36]:


#2 Display the last number in the list
print(numbers[-1])


# In[37]:


#3 Command for displaying the values from the list [0:4]
print(numbers[0:4])


# In[38]:


#4 Command for displaying the values from the list [2:]
print(numbers[2:])


# In[39]:


#5 Command for displaying the values from the list [:6]
print(numbers[:6])


# # 2 Manipulate using tuples

# In[52]:


#1 Create a tuple
my_tuple = (2,4,6,8)
print("Tuple:", my_tuple)


# In[54]:


#2 Add new elements to the end of the tuple
new = 5
my_tuple += (new,)
print("Tuple:", my_tuple)


# In[55]:


#3 Reverse elements in the tuple
reversed_tuple = tuple(reversed(my_tuple))
print("Reversed Tuple:", reversed_tuple)



# In[60]:


#4 Display elements of the tuple multiple times
repeat_times = 5
repeated_tuple = my_tuple * repeat_times
print("Repeated Tuple:", repeated_tuple)



# In[59]:


#5 Concatenate two tuples
tuple1 = (1, 9, 5)
tuple2 = (2, 4, 6)
concatenated_tuple = tuple1 + tuple2
print("Concatenated Tuple:", concatenated_tuple)


# In[57]:


#6 Sort the elements in the tuple in ascending order
sorted_tuple = tuple(sorted(my_tuple))
print("Sorted Tuple:", sorted_tuple)


# # 5 Python program for strings

# In[61]:


#1)to calculate the length of string
string='Good Morning'
len(string)


# In[63]:


#2) reverse word in string
string='Mphasis'
string[::-1]


# In[70]:


#3)to display the same string multiple times

str = string* 5
print(str)


# In[71]:


#4)to concatenate two strings
str1='welcomes you'
a=string +" "+str1
print(a)


# In[72]:


#5) str1="south India",using string slicing to display"India"
str1='South India'
str1[6:11]


# # 8 to delete an element from a list to index

# In[78]:


list = [6,3,55,3,2,7,99,65,90]
del list[5]
print(list)


# # 9 to display a number from 1 to 100

# In[75]:


i= 1
while i <= 100:
    print(i, end=" ")
    i+=1


# # 10 to find sum of all items in tuple

# In[77]:


tuple = (45,83,23,11,99)
result = sum(tuple)
print(result)


# # 11 Lambda Functions

# In[83]:


# Define the dictionary with lambda functions
func_dict = {
    'Square': lambda x: x**2,
    'Cube': lambda x: x**3,
    'Squareroot': lambda x: x**0.5
}


# In[84]:


# Get user input for a number
num = float(input("Enter a number: "))


# In[ ]:


# Initialize a variable to store the sum of outputs
result = 0


# In[ ]:


# Iterate through the functions in the dictionary, apply them, and add to the result
for func_name, func in func_dict.items():
    output = func(num)
    result += output


# In[ ]:


# Print the sum of outputs
print("Sum of outputs:", result)


# # 12. A list of words is given. Find the words from the list that have their second character in uppercase. ['hello', 'Dear', 'how', 'ARe', 'You']

# In[85]:


word_list = ['hello', 'Dear', 'how', 'ARe', 'You']
result = []

for word in word_list:
    if len(word) > 1 and word[1].isupper():
        result.append(word)

print(result)


# # 6. Perform the following in dictionary

# In[103]:


my_dict = {
    "key1": "value1",
    "key2": "value2",
    "key3": "value3"
}
# Accessing values by key
value = my_dict["key1"]
print(value)  # This will print "value1"


# In[102]:


print(my_dict) 
my_dict.clear()
del my_dict["key1"]


# # 7. insert any position in a list

# In[107]:


print("Enter 10 Elements of List: ")
nums = []
for i in range(10):
    nums.insert(i, input())
print("Enter an Element to Insert at End: ")
elem = input()
nums.append(elem)
print("\nThe New List is: ")
print(nums)


# #  2.CONTROL STRUCTURES

# # 1.PROGRAM TO FIND THE FIRST N PRIME NUMBERS

# In[109]:


def is_prime(num):
    if num <= 1:
        return False
    elif num <= 3:
        return True
    elif num % 2 == 0 or num % 3 == 0:
        return False
    i = 5
    while i * i <= num:
        if num % i == 0 or num % (i + 2) == 0:
            return False
        i += 6
    return True

def first_n_primes(n):
    primes = []
    num = 2
    while len(primes) < n:
        if is_prime(num):
            primes.append(num)
        num += 1
    return primes
n = int(input("Enter the value of n: "))
prime_numbers = first_n_primes(n)
print(f"The first {n} prime numbers are: {prime_numbers}")


# # 2. Write the python code that calculates the salary of an employee. Prompt the user to enter the Basic Salary, HRA, TA, and DA. Add these components to calculate the Gross Salary. Also, deduct 10% of salary from the Gross Salary to be paid as tax and display gross minus tax as net salary.
# 

# In[110]:


# Prompt the user to enter Basic Salary, HRA, TA, and DA
basic_salary = float(input("Enter Basic Salary: "))
hra = float(input("Enter HRA: "))
ta = float(input("Enter TA: "))
da = float(input("Enter DA: "))

# Calculate Gross Salary by adding Basic, HRA, TA, and DA
gross_salary = basic_salary + hra + ta + da

# Calculate Tax (10% of Gross Salary)
tax = 0.10 * gross_salary

# Calculate Net Salary (Gross Salary - Tax)
net_salary = gross_salary - tax

# Display Gross and Net Salary
print(f"Gross Salary: {gross_salary}")
print(f"Net Salary: {net_salary}")


# # 3.search for a string in a given list

# In[114]:


def search_string_in_list(search_string, my_list):
    found_indices = []
    for i, item in enumerate(my_list):
        if search_string in item:
            found_indices.append(i)
    
    if found_indices:
        print(f"'{search_string}' found in the list at indices: {found_indices}")
    else:
        print(f"'{search_string}' not found in the list.")
        
# Example usage:
my_list = ["Kishore", "Pavan", "Govind", "Darshan", "Lohith"]
search_string = "Govind"
search_string_in_list(search_string, my_list)


# # 4. Write a Python function that accepts a string and calculates the number of upper-case letters and lower-case letters.

# In[116]:


def count_case_letters(input_string):
    upper_count = 0
    lower_count = 0
    
    for char in input_string:
        if char.isupper():
            upper_count += 1
        elif char.islower():
            lower_count += 1
    
    return upper_count, lower_count
input_str = "Mphasis Learning Academy"
upper, lower = count_case_letters(input_str)
print("Uppercase letters:", upper)
print("Lowercase letters:", lower)


# # 5. Write a program to display the sum of odd numbers and even numbers that fall between 12 and 37.

# In[117]:


sum_odd = 0
sum_even = 0
for num in range(12, 38):
    if num % 2 == 0:
        sum_even += num
    else:
        sum_odd += num
print("Sum of even numbers:", sum_even)
print("Sum of odd numbers:", sum_odd)


# # 6.print the table of any number

# In[118]:


# Get the number from the user
num = int(input("Enter a number: "))

# Print the multiplication table
print(f"Multiplication Table of {num}:")
for i in range(1, 11):
    print(f"{num} x {i} = {num * i}")


# # 7.sum the first 10 prime numbers

# In[119]:


def is_prime(num):
    if num < 2:
        return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    return True

count = 0
sum_of_primes = 0
number = 2

while count < 10:
    if is_prime(number):
        sum_of_primes += number
        count += 1
    number += 1

print(f"The sum of the first 10 prime numbers is: {sum_of_primes}")


# # 8.You can implement arithmetic operations using nested if statements in Python like this:

# In[120]:


# Get two numbers from the user
num1 = float(input("Enter the first number: "))
num2 = float(input("Enter the second number: "))

# Get the desired operation from the user
operation = input("Enter an arithmetic operation (+, -, *, /): ")

# Perform the selected operation
if operation == "+":
    result = num1 + num2
elif operation == "-":
    result = num1 - num2
elif operation == "*":
    result = num1 * num2
elif operation == "/":
    if num2 != 0:
        result = num1 / num2
    else:
        result = "Division by zero is not allowed."
else:
    result = "Invalid operation"

print(f"Result: {result}")


# # 9.The temperature in celsius and convert it to a Fahrenheit

# In[121]:


# Input temperature in Celsius
celsius = float(input("Enter temperature in Celsius: "))

# Convert to Fahrenheit
fahrenheit = (celsius * 9/5) + 32

# Display the result
print(f"{celsius} degrees Celsius is equal to {fahrenheit} degrees Fahrenheit")


# # 10.Find a maximum and minimum number in a list without using an inbuilt function

# In[126]:


# Function to find maximum and minimum in a list
def find_max_min(numbers):
    # Check if the list is empty
    if not numbers:
        return None, None

    # Initialize variables to store maximum and minimum
    maximum = minimum = numbers[0]

    # Iterate through the list
    for number in numbers:
        if number > maximum:
            maximum = number
        if number < minimum:
            minimum = number

    return maximum, minimum

# Example usage
numbers = [3,5,2,1,9,6,8,0,7]
max_num, min_num = find_max_min(numbers)
print(f"Maximum: {max_num}")
print(f"Minimum: {min_num}")


# # 11.Write a program in python to print out the number of seconds in 30-day month 30 days, 24 hours in a day, 60 minutes per day, 60 seconds in a minute

# In[128]:


days_in_month = 30
hours_in_day = 24
minutes_in_hour = 60
seconds_in_minute = 60

seconds_in_30_days = days_in_month * hours_in_day * minutes_in_hour * seconds_in_minute

print(f"There are {seconds_in_30_days} seconds in a 30-day month.")


# # 12.printout the number of seconds in a year

# In[129]:


# Constants for the number of days, hours, minutes, and seconds
days_per_year = 365
hours_per_day = 24
minutes_per_hour = 60
seconds_per_minute = 60

# Calculate the total number of seconds in a year
total_seconds = days_per_year * hours_per_day * minutes_per_hour * seconds_per_minute

# Display the result
print(f"The number of seconds in a year (assuming 365 days) is: {total_seconds} seconds")


# # 13. A high-speed train can travel at an average speed of 150 mph, how long will it take a train travelling at this speed to travel from London to Glasgow which is 414 miles
# 

# In[130]:


# Define the distance in miles and the average speed in mph
distance = 414
speed = 150

# Calculate the time in hours
time_hours = distance / speed

# Convert hours to hours and minutes
hours = int(time_hours)
minutes = (time_hours - hours) * 60

# Print the result
print(f"It will take approximately {hours} hours and {minutes:.2f} minutes to travel from London to Glasgow.")# Define the distance in miles and the average speed in mph
distance = 414
speed = 150

# Calculate the time in hours
time_hours = distance / speed

# Convert hours to hours and minutes
hours = int(time_hours)
minutes = (time_hours - hours) * 60

# Print the result
print(f"It will take approximately {hours} hours and {minutes:.2f} minutes to travel from London to Glasgow.")


# # 14.School Python Program

# In[ ]:


# Define the variable days_in_each_school_year
days_in_each_school_year = 192

# Years 7 to 11
years = range(7, 12)

# Calculate the total hours spent in school
total_hours = sum(year * days_in_each_school_year * 6 for year in years)

# Display the result
print(f"Total hours spent in school from year 7 to year 11: {total_hours} hours")


# # 15. If the age of Ram,Sam and Khan are input through the keyboard, write a python program to determine the eldest and youngest of the three

# In[132]:


# Input ages of Ram, Sam, and Khan
ram_age = int(input("Enter Ram's age: "))
sam_age = int(input("Enter Sam's age: "))
khan_age = int(input("Enter Khan's age: "))

# Determine the eldest and youngest
if ram_age >= sam_age and ram_age >= khan_age:
    eldest = "Ram"
    if sam_age <= khan_age:
        youngest = "Sam"
    else:
        youngest = "Khan"
elif sam_age >= ram_age and sam_age >= khan_age:
    eldest = "Sam"
    if ram_age <= khan_age:
        youngest = "Ram"
    else:
        youngest = "Khan"
else:
    eldest = "Khan"
    if ram_age <= sam_age:
        youngest = "Ram"
    else:
        youngest = "Sam"

# Print the results
print(f"The eldest among Ram, Sam, and Khan is: {eldest}")
print(f"The youngest among Ram, Sam, and Khan is: {youngest}")


# # 16.with nd without slicing

# In[133]:


def rotate_list_using_slicing(input_list, n):
    if len(input_list) == 0:
        return input_list
    
    n %= len(input_list)  # Ensure n is within the length of the list
    rotated_list = input_list[-n:] + input_list[:-n]
    return rotated_list

# Input list
my_list = [1, 2, 3, 4, 5]
n = int(input("Enter the number of times to rotate to the right: "))

rotated_list = rotate_list_using_slicing(my_list, n)
print("Rotated list using slicing technique:", rotated_list)


# In[134]:


def rotate_list_without_slicing(input_list, n):
    if len(input_list) == 0:
        return input_list
    
    n %= len(input_list)  # Ensure n is within the length of the list
    for _ in range(n):
        temp = input_list.pop()
        input_list.insert(0, temp)
    return input_list

# Input list
my_list = [1, 2, 3, 4, 5]
n = int(input("Enter the number of times to rotate to the right: "))

rotated_list = rotate_list_without_slicing(my_list, n)
print("Rotated list without slicing technique:", rotated_list)


# #    17.print the patterns given below

# In[146]:


1. # Input the number of rows for the pattern
n = int(input("Enter the no of rows: "))

# Function to calculate binomial coefficients
def binomial_coefficient(n, k):
    if k == 0 or k == n:
        return 1
    return binomial_coefficient(n - 1, k - 1) + binomial_coefficient(n - 1, k)

# Loop to print the pattern
for i in range(n):
    for j in range(i + 1):
        print(binomial_coefficient(i, j), end=" ")
    print()


# 

# 

# In[147]:


# 2.Pattern program 
n = 5  
for i in range(n):
    for j in range(i + 1):
        print("*", end=" ")
    print() 


# In[143]:


3 # Input the number of rows for the pattern
n = int(input("Enter the number of rows: "))

# Loop to print the pattern
for i in range(1, n + 1):
    print(" " * (n - i), end="")  # Print spaces before asterisks
    print("* " * i)  # Print asterisks with a space in between


# In[145]:


#4.Python Word pattern
word = "Python"
for i in range(len(word) + 1):
    print(word[:i])
print("Python")


# 

# 

# 

# 

# 

# 

# 

# 

# 

# 
