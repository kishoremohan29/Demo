#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 1.Write a Python function to list even and odd numbers in a list


# In[ ]:


odd=[]
even=[]
def divide():
    for i in lst:
     if i%2==0:
        even.append(i)
     else:
        odd.append(i)
        
divide()       
print('Even numbers in given list are:',even)
print('Odd numbers in given list are:',odd)



# In[ ]:


# 2 Write and run python program ask user to enter 8 integers(one at a time) and then print out how many those integers were even numbers. Eg, if user enter 19,6,9,20,7,6,and 1 then program should print out 3 since 3 of those numbers are even


# In[ ]:


lst=[]
for i in range(8):
    lst.append(int(input("enter the number:")))
    
def even():
    count=0
    for x in lst:
        if x%2==0:
            count+=1
    return count
even()


# In[ ]:


# 3 Write a python program where you take any positive integer n,if n is even, divide it by 2 to get n/2 if n is odd, multiply it by 3 and add 2 to obtain 3n+1. Repeat the process until you reach 1


# In[ ]:


n=int(input("enter the number:"))

def get(n):
    while n!=1:
        if n%2==0:
            n=n/2
        else:
            n=3*n+1
    return n
get(n)



# In[ ]:


# 4 Write a python program to compute sum of all the multiples of 3 or 5 below 500


# In[ ]:


def sumfun():
    sum=0
    for i in range(1,500):
        if i%3==0 or i%5==0:
            sum+=i
    return sum

sumfun()



# In[ ]:


# 6  Write python program to compute matrix multiplication


# In[ ]:


import numpy as np

A = [[12, 7, 3],[4, 5, 6],[7, 8, 9]]
 

B = [[5, 8, 1, 2],[6, 7, 3, 0],[4, 5, 9, 1]]
 
result = np.dot(A,B)
print(result)
     


# In[ ]:


# 7 Write a python function to count the no of vowels in a string


# In[ ]:


def vow(s):
    count=0
    vow=['a','e','i','o','u']
    for i in s:
        if i in vow:
            count+=1
    return count

s=input()
vow(s)


# In[ ]:


# 8 Write a function for finding factorial for given number using recursive function.


# In[ ]:


def fact(n):
    if n==0 or n==1:
        return 1
    else:
        return n*fact(n-1)

n=int(input("enter the number:"))
fact(n)


# In[ ]:


# 9  Write a python function for generating fibonacci series using function


# In[ ]:


def fib(n):
    a, b = 0, 1
    for i in range(n):
        a, b = b, a + b
    return a

        
n=int(input("enter the number:"))
fib(n) 


# In[ ]:


# 10 Write a python program to display given integer in reverse order using function without in-built functions


# In[ ]:


def rev(n):
    rev_num=06
    while n > 0:
        rev_num = rev_num * 10 + n % 10
        n //= 10
    return rev_num
n=int(input("enter the number:"))
rev(n)



# In[ ]:


# 11 Write a python function to display all the integers within the range 200-300 whose sum of digit is an even number


# In[ ]:


def evensum():
    for n in range(200,300):
        sum=0
        for i in str(n):
               sum+=int(i)
        if sum%2==0:
            print(n)
            
evensum()


# In[ ]:


# 12 Write a python function to find the number of digits and sum of digits of given integer


# In[ ]:


def dig(n):
    num = 0
    sum = 0
    while n > 0:
        num += 1
        sum+= n % 10
        n //= 10
    return num, sum


n=int(input("enter the number:"))
num,sum=dig(n)
print('the number of digits are:',num)
print('the sum of digits is:',sum)


# In[ ]:


#  13 Write function called is_sorted that takes a list as a parameters and returns true if the list is sorted in ascending order and false otherwise and has duplicate that takes a list and returns true if there is any element that appears more than once.it shold not modify the original list.



# In[ ]:


def is_sorted(arr):
    # Check if the list is sorted in ascending order.
    return all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1))

def has_duplicates(arr):
    # Create a set to store unique elements.
    seen = set()
    
    for element in arr:
        if element in seen:
            return True
        seen.add(element)
    
    return False

# Example usage:
my_list = [1, 2, 3, 4, 5]
print("is_sorted(my_list):", is_sorted(my_list))  # True

my_list_with_duplicates = [1, 2, 2, 3, 4]
print("has_duplicates(my_list_with_duplicates):", has_duplicates(my_list_with_duplicates))


# In[ ]:


# 14 Write a function called nested_sum that takes a list of integers and add up the elements from all the nested lists and cumsum that takes a list of numbers and returns the cummulative sum;that is a new list where the ith element is the sum of the first i+1 elements from the original list.


# In[ ]:


def nested_sum(nested_list):
    total = 0
    for sublist in nested_list:
        for element in sublist:
            if isinstance(element, int):
                total += element    
    return total
def cumsum(numbers):
    cumulative_sum = 0
    cumsum_list = []
    for num in numbers:
        cumulative_sum += num
        cumsum_list.append(cumulative_sum)
    return cumsum_list

nested_list = [[1, 2, 3], [4, 5], [6, 7, 8]]
print("nested_sum(nested_list):", nested_sum(nested_list))  

numbers = [1, 2, 3, 4, 5]
print("cumsum(numbers):", cumsum(numbers)) 


# In[ ]:





# In[ ]:





# In[ ]:




