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


# In[ ]:





# In[ ]:




