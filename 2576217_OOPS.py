#!/usr/bin/env python
# coding: utf-8

# # 1. Write a python program to create a base class "Shape" with methods to calculate area and perimeter. Then, create derived classes "Circle" and "Rectangle" that inherit from the base class and calculate their respective areas and perimeters. Demonstrate their usage in a program.
# 
# # You are developing an online quiz application where users can take quizzes on various topics and receive scores.
# 
# # 1. Create a class for quizzes and questions.
# 
# # 2. Implement a scoring system that calculates the user's score on a quiz.
# 
# # 3. How would you store and retrieve user progress, including quiz history and scores?
# 

# In[1]:


import math

class Shape:
    def calculate_area(self):
        pass

    def calculate_perimeter(self):
        pass

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def calculate_area(self):
        return math.pi * self.radius ** 2

    def calculate_perimeter(self):
        return 2 * math.pi * self.radius

class Rectangle(Shape):
    def __init__(self, length, width):
        self.length = length
        self.width = width

    def calculate_area(self):
        return self.length * self.width

    def calculate_perimeter(self):
        return 2 * (self.length + self.width)

# Demonstrate usage
circle = Circle(3)
rectangle = Rectangle(6, 8)

print(f"Circle - Area: {circle.calculate_area()}, Perimeter: {circle.calculate_perimeter()}")
print(f"Rectangle - Area: {rectangle.calculate_area()}, Perimeter: {rectangle.calculate_perimeter()}")

class Question:
    def __init__(self, text, correct_answer):
        self.text = text
        self.correct_answer = correct_answer

    def check_answer(self, user_answer):
        return user_answer == self.correct_answer

class Quiz:
    def __init__(self, name, questions):
        self.name = name
        self.questions = questions

    def take_quiz(self):
        score = 0
        for question in self.questions:
            user_answer = input(question.text + " ")
            if question.check_answer(user_answer):
                score += 1
        print(f"Your score for {self.name}: {score}/{len(self.questions)}")

# Example usage
question1 = Question("Who is India's Prime Minister?", "Narendra Modi")
question2 = Question("What is the capital of Tamil Nadu?", "Chennai")
quiz = Quiz("General Knowledge Quiz", [question1, question2])
quiz.take_quiz()


# # 2. Write a python script to create a class "Person" with private attributes for age and name. Implement a method to calculate a person's eligibility for voting based on their age. Ensure that age cannot be accessed directly but only through a getter method

# In[2]:


class Person:
    def __init__(self, name, age):
        self.__name = name
        self.__age = age

    def get_name(self):
        return self.__name

    def get_age(self):
        return self.__age

    def is_eligible_to_vote(self):
        if self.__age >= 18:
            return True
        else:
            return False

# Example usage:
person1 = Person("Govind", 32)
print(f"{person1.get_name()} is eligible to vote: {person1.is_eligible_to_vote()}")


# # 3

# In[4]:


class BankAccount:
    def __init__(self, account_number, account_holder_name, balance=0):
        self.__account_number = account_number
        self.__account_holder_name = account_holder_name
        self.__balance = balance

    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount
            print(f"Deposited Rs.{amount}. New balance: Rs.{self.__balance}")
        else:
            print("Invalid deposit amount.")

    def withdraw(self, amount):
        if amount > 0 and amount <= self.__balance:
            self.__balance -= amount
            print(f"Withdrew Rs.{amount}. New balance: Rs.{self.__balance}")
        else:
            print("Invalid withdrawal amount or insufficient funds.")

    def get_balance(self):
        return self.__balance

    def get_account_number(self):
        return self.__account_number


class SavingsAccount(BankAccount):
    def __init__(self, account_number, account_holder_name, balance=0, interest_rate=0.02):
        super().__init__(account_number, account_holder_name, balance)
        self.__interest_rate = interest_rate

    def calculate_interest(self):
        interest = self.get_balance() * self.__interest_rate
        self.deposit(interest)
        print(f"Interest of Rs.{interest} added to the account.")


class CheckingAccount(BankAccount):
    def __init__(self, account_number, account_holder_name, balance=0, overdraft_limit=100):
        super().__init__(account_number, account_holder_name, balance)
        self.__overdraft_limit = overdraft_limit

    def withdraw(self, amount):
        if amount <= self.get_balance() + self.__overdraft_limit:
            super().withdraw(amount)
        else:
            print("Withdrawal exceeds overdraft limit.")


# Usage example:
if __name__ == "__main__":
    savings_account = SavingsAccount("77087", "Kishore Mohan", 2000, 0.08)
    checking_account = CheckingAccount("27983", "Pavan Hebbar", 30000, 200)

    savings_account.deposit(500)
    savings_account.calculate_interest()
    savings_account.withdraw(200)

    checking_account.withdraw(700)


# # 4

# In[10]:


class Employee:
    def __init__(self, name, employee_id, salary):
        self.__name = name
        self.__employee_id = employee_id
        self.__salary = salary

    # Getter methods
    def get_name(self):
        return self.__name

    def get_employee_id(self):
        return self.__employee_id

    def get_salary(self):
        return self.__salary

    # Setter methods
    def set_name(self, name):
        self.__name = name

    def set_employee_id(self, employee_id):
        self.__employee_id = employee_id

    def set_salary(self, salary):
        self.__salary = salary

    # Salary calculation method (to be overridden by subclasses)
    def calculate_salary(self):
        pass

# Step 2: Create subclasses "FullTimeEmployee" and "PartTimeEmployee"
class FullTimeEmployee(Employee):
    def __init__(self, name, employee_id, annual_salary):
        super().__init__(name, employee_id, annual_salary)

    # Override the salary calculation method
    def calculate_salary(self):
        return self.get_salary()

class PartTimeEmployee(Employee):
    def __init__(self, name, employee_id, hours_worked, hourly_rate):
        super().__init__(name, employee_id, 0)  # Initialize salary to 0 for part-time employees
        self.__hours_worked = hours_worked
        self.__hourly_rate = hourly_rate

    # Getter method for hours worked
    def get_hours_worked(self):
        return self.__hours_worked

    # Setter method for hours worked
    def set_hours_worked(self, hours_worked):
        self.__hours_worked = hours_worked

    # Override the salary calculation method
    def calculate_salary(self):
        return self.__hours_worked * self.__hourly_rate

# Step 4: Demonstrate polymorphism
full_time_employee = FullTimeEmployee("Kishore Mohan", 203, 50000)
part_time_employee = PartTimeEmployee("Darshan H", 105, 20, 1800)

employees = [full_time_employee, part_time_employee]

for employee in employees:
    print(f"Employee Name: {employee.get_name()}")
    print(f"Employee ID: {employee.get_employee_id()}")
    print(f"Salary: Rs.{employee.calculate_salary()}")
    print()


# # 5. Library Management System-Scenario

# In[12]:


import datetime

class Book:
    def __init__(self, title, author, publication_date, isbn):
        self.__title = title
        self.__author = author
        self.__publication_date = publication_date
        self.__isbn = isbn
        self.__checked_out = False

    def get_title(self):
        return self.__title

    def get_author(self):
        return self.__author

    def get_publication_date(self):
        return self.__publication_date

    def get_isbn(self):
        return self.__isbn

    def is_checked_out(self):
        return self.__checked_out

    def check_out(self):
        self.__checked_out = True

    def return_book(self):
        self.__checked_out = False

class FictionBook(Book):
    def __init__(self, title, author, publication_date, isbn, genre):
        super().__init__(title, author, publication_date, isbn)
        self.__genre = genre

    def get_genre(self):
        return self.__genre

class NonFictionBook(Book):
    def __init__(self, title, author, publication_date, isbn, topic):
        super().__init__(title, author, publication_date, isbn)
        self.__topic = topic

    def get_topic(self):
        return self.__topic

class Patron:
    def __init__(self, name, patron_id):
        self.__name = name
        self.__patron_id = patron_id

    def get_name(self):
        return self.__name

    def get_patron_id(self):
        return self.__patron_id

class Transaction:
    def __init__(self, book, patron):
        self.__book = book
        self.__patron = patron
        self.__checkout_date = datetime.date.today()
        self.__due_date = self.__checkout_date + datetime.timedelta(days=14)

    def get_book(self):
        return self.__book

    def get_patron(self):
        return self.__patron

    def get_checkout_date(self):
        return self.__checkout_date

    def get_due_date(self):
        return self.__due_date

    def is_overdue(self):
        return datetime.date.today() > self.__due_date


if __name__ == "__main__":
    fiction_book = FictionBook("The Alchemist", " Paulo Coelho", "1987", "978-0743273565", "Psychology")
    nonfiction_book = NonFictionBook("Untamed", "Glennon Doyle", "2016", "978-0123456097", "Social Science")
    patron = Patron("Alice", "12345")

    transaction1 = Transaction(fiction_book, patron)
    transaction2 = Transaction(nonfiction_book, patron)

    fiction_book.check_out()
    print(fiction_book.is_checked_out())  # Output: True

    nonfiction_book.return_book()
    print(nonfiction_book.is_checked_out())  # Output: False

    print(transaction1.is_overdue())  # Output: False
    print(transaction2.is_overdue())  # Output: True


# # 6.Online Shopping Cart

# In[16]:


class Product:
    def __init__(self, product_id, name, price):
        self.product_id = product_id
        self.name = name
        self.price = price

    def get_product_id(self):
        return self.product_id

    def get_name(self):
        return self.name

    def get_price(self):
        return self.price

class ShoppingCart:
    def __init__(self):
        self.items = []

    def add_product(self, product, quantity=1):
        self.items.append({"product": product, "quantity": quantity})

    def remove_product(self, product):
        self.items = [item for item in self.items if item["product"] != product]

    def calculate_total_cost(self):
        total_cost = 0
        for item in self.items:
            total_cost += item["product"].get_price() * item["quantity"]
        return total_cost

class Order:
    def __init__(self, order_id, customer):
        self.order_id = order_id
        self.customer = customer
        self.items = []

    def add_item(self, product, quantity=1):
        self.items.append({"product": product, "quantity": quantity})

    def calculate_order_total(self):
        total_cost = 0
        for item in self.items:
            total_cost += item["product"].get_price() * item["quantity"]
        return total_cost

# Inheritance example
class Electronics(Product):
    def __init__(self, product_id, name, price, brand):
        super().__init__(product_id, name, price)
        self.brand = brand

# Usage
iphone = Electronics("1", "iPhone 15 Pro", 140000, "Apple")
laptop = Electronics("2", "Macbook Air", .130000, "Apple")

cart = ShoppingCart()
cart.add_product(iphone, 2)
cart.add_product(laptop)

order = Order("246", "Kishore Mohan")
order.add_item(iphone, 4)
order.add_item(laptop)

total_cart_cost = cart.calculate_total_cost()
total_order_cost = order.calculate_order_total()

print(f"Total Cart Cost: Rs.{total_cart_cost}")
print(f"Total Order Cost: Rs.{total_order_cost}")


# In[ ]:





# In[ ]:




