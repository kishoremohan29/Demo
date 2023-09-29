#!/usr/bin/env python
# coding: utf-8

# In[9]:


import subprocess
import sys
import os

def create_virtual_environment(project_name):
    try:
        # Create a virtual environment for the specified project
        subprocess.run([sys.executable, '-m', 'venv', project_name])
        print(f"Virtual environment '{project_name}' created successfully.")
    except Exception as e:
        print(f"Error creating virtual environment: {e}")

def activate_virtual_environment(venv_path):
    try:
        # Activate the virtual environment
        if sys.platform == 'win32':
            activate_script = os.path.join(venv_path, 'Scripts', 'activate')
        else:
            activate_script = os.path.join(venv_path, 'bin', 'activate')

        activate_cmd = f'source {activate_script}'
        subprocess.run(activate_cmd, shell=True)
        print(f"Activated virtual environment at '{venv_path}'.")
    except Exception as e:
        print(f"Error activating virtual environment: {e}")

def deactivate_virtual_environment():
    try:
        # Deactivate the virtual environment
        subprocess.run(['deactivate'], shell=True)
        print("Deactivated virtual environment.")
    except Exception as e:
        print(f"Error deactivating virtual environment: {e}")

def install_package(venv_path, package_name):
    try:
        # Install a package in the virtual environment
        subprocess.run([os.path.join(venv_path, 'bin', 'pip'), 'install', package_name])
        print(f"Package '{package_name}' installed in the virtual environment.")
    except Exception as e:
        print(f"Error installing package: {e}")

def list_installed_packages(venv_path):
    try:
        # List installed packages in the virtual environment
        subprocess.run([os.path.join(venv_path, 'bin', 'pip'), 'freeze'])
    except Exception as e:
        print(f"Error listing installed packages: {e}")

def main():
    project_name = 'my_project'
    create_virtual_environment(project_name)

    # Specify the virtual environment path
    venv_path = os.path.join(os.getcwd(), project_name)

    activate_virtual_environment(venv_path)

    # Install or upgrade packages within the virtual environment
    install_package(venv_path, 'requests')
    install_package(venv_path, '--upgrade pandas')

    # List installed packages
    list_installed_packages(venv_path)

    # Deactivate the virtual environment
    deactivate_virtual_environment()

if __name__== "__main__":
    main()


# In[ ]:


#2


# In[3]:


import subprocess

def install_dependencies(requirements_file):
    try:
        # Execute the 'pip install' command to install dependencies
        subprocess.run(['pip', 'install', '-r', requirements_file], check=True)
        print("Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during dependency installation (Exit Code {e.returncode}):")
        print(e.stderr.decode())
    except FileNotFoundError:
        print("Error: 'pip' command not found. Please make sure pip is installed.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    # Specify the path to your requirements.txt file
    requirements_file = 'requirements.txt'

    # Call the function to install dependencies
    install_dependencies(requirements_file)

if __name__ == "__main__":
    main()


# In[ ]:


#3


# In[4]:


import mysql.connector

# Establish a connection to the MySQL database
conn = mysql.connector.connect(
    host="127.0.0.1",
    user="root",
    password="Correctpassword@2903",
    database="inventory_db"
)

cursor = conn.cursor()

def add_product(name, description, price, quantity):
    query = "INSERT INTO products (name, description, price, quantity) VALUES (%s, %s, %s, %s)"
    data = (name, description, price, quantity)
    cursor.execute(query, data)
    conn.commit()
    print("Product added successfully!")

def update_product(product_id, quantity_change):
    query = "UPDATE products SET quantity = quantity + %s WHERE id = %s"
    data = (quantity_change, product_id)
    cursor.execute(query, data)
    conn.commit()
    print("Inventory updated successfully!")

def view_products():
    query = "SELECT * FROM products"
    cursor.execute(query)
    products = cursor.fetchall()
    if not products:
        print("No products found.")
    else:
        for product in products:
            print(product)

def delete_product(product_id):
    query = "DELETE FROM products WHERE id = %s"
    data = (product_id,)
    cursor.execute(query, data)
    conn.commit()
    print("Product deleted successfully!")

while True:
    print("\nInventory Management System")
    print("1. Add Product")
    print("2. Update Inventory")
    print("3. View Products")
    print("4. Delete Product")
    print("5. Quit")

    choice = input("Enter your choice: ")

    if choice == "1":
        name = input("Enter product name: ")
        description = input("Enter product description: ")
        price = float(input("Enter product price: "))
        quantity = int(input("Enter product quantity: "))
        add_product(name, description, price, quantity)
    elif choice == "2":
        product_id = int(input("Enter product ID to update inventory: "))
        quantity_change = int(input("Enter quantity change: "))
        update_product(product_id, quantity_change)
    elif choice == "3":
        view_products()
    elif choice == "4":
        product_id = int(input("Enter product ID to delete: "))
        delete_product(product_id)
    elif choice == "5":
        break
    else:
        print("Invalid choice. Please try again.")

# Close the cursor and database connection
cursor.close()
conn.close()


# In[ ]:


#1


# In[5]:


import mysql.connector
from datetime import date

# Establish a connection to the MySQL database
conn = mysql.connector.connect(
    host="127.0.0.1",
    user="root",
    password="Correctpassword@2903",
    database="order_processing_db"
)

cursor = conn.cursor()

def place_order(customer_id, product_id, quantity):
    try:
        # Check if the product is available in sufficient quantity
        cursor.execute("SELECT price, quantity_available FROM products WHERE product_id = %s", (product_id,))
        result = cursor.fetchone()

        if result is None:
            print("Product not found.")
            return
        elif result[1] < quantity:
            print("Insufficient quantity available.")
            return

        # Calculate the subtotal
        subtotal = result[0] * quantity

        # Insert into orders table
        cursor.execute("INSERT INTO orders (customer_id, order_date, total_cost) VALUES (%s, %s, %s)",
                       (customer_id, date.today(), subtotal))
        order_id = cursor.lastrowid

        # Insert into order_details table
        cursor.execute("INSERT INTO order_details (order_id, product_id, quantity_ordered, subtotal) VALUES (%s, %s, %s, %s)",
                       (order_id, product_id, quantity, subtotal))

        # Update product quantity
        cursor.execute("UPDATE products SET quantity_available = quantity_available - %s WHERE product_id = %s",
                       (quantity, product_id))

        conn.commit()
        print("Order placed successfully!")

    except Exception as e:
        conn.rollback()
        print(f"Error placing order: {e}")

# Example usage:
customer_id = 1  # Replace with the actual customer ID
product_id = 1   # Replace with the actual product ID
quantity = 3

place_order(customer_id, product_id, quantity)

# Close the cursor and database connection
cursor.close()
conn.close()


# In[6]:


import mysql.connector
from datetime import date

# Establish a connection to the MySQL database
conn = mysql.connector.connect(
    host="127.0.0.1",
    user="root",
    password="Correctpassword@2903",
    database="order_processing_db"
)

cursor = conn.cursor()

def place_order(customer_id, product_id, quantity):
    try:
        # Check if the product is available in sufficient quantity
        cursor.execute("SELECT price, quantity_available FROM products WHERE product_id = %s", (product_id,))
        result = cursor.fetchone()

        if result is None:
            print("Product not found.")
            return
        elif result[1] < quantity:
            print("Insufficient quantity available.")
            return

        # Calculate the subtotal
        subtotal = result[0] * quantity

        # Insert into orders table
        cursor.execute("INSERT INTO orders (customer_id, order_date, total_cost) VALUES (%s, %s, %s)",
                       (customer_id, date.today(), subtotal))
        order_id = cursor.lastrowid

        # Insert into order_details table
        cursor.execute("INSERT INTO order_details (order_id, product_id, quantity_ordered, subtotal) VALUES (%s, %s, %s, %s)",
                       (order_id, product_id, quantity, subtotal))

        # Update product quantity
        cursor.execute("UPDATE products SET quantity_available = quantity_available - %s WHERE product_id = %s",
                       (quantity, product_id))

        conn.commit()
        print("Order placed successfully!")

    except Exception as e:
        conn.rollback()
        print(f"Error placing order: {e}")

# Example usage:
customer_id = 1  # Replace with the actual customer ID
product_id = 1   # Replace with the actual product ID
quantity = 3

place_order(customer_id, product_id, quantity)

# Close the cursor and database connection
cursor.close()
conn.close()


# In[ ]:


#2


# In[7]:


import mysql.connector

# Define the database connection parameters
db_config = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': 'Ramyasaki@18',
    'database': 'yourdatabasename'
}

try:
    # Connect to the MySQL database
    connection = mysql.connector.connect(**db_config)

    # Create a cursor object to interact with the database
    cursor = connection.cursor()

    # Step 2(ii): Retrieve all records from the table
    cursor.execute("SELECT * FROM orderdetails")
    records = cursor.fetchall()

    # Initialize a variable to store the total quantity
    total_quantity = 0

    # Step 2(iii): Calculate the total quantity
    for record in records:
        total_quantity += record[2]

    # Step 2(iv): Update the quantity column by doubling its value
    for record in records:
        new_quantity = record[2] * 2
        cursor.execute("UPDATE your_table SET quantity = %s WHERE id = %s", (new_quantity, record[0]))

    # Commit the changes to the database
    connection.commit()

    # Step 2(vi): Close the database connection
    cursor.close()
    connection.close()

    print("Operation completed successfully.")
except mysql.connector.Error as err:
    print(f"Error: {err}")
finally:
    if connection.is_connected():
        cursor.close()
        connection.close()


# In[ ]:


#3


# In[8]:


import mysql.connector

# Define the database connection parameters
db_config = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'root',
    'password': 'Ramyasaki@18',
    'database': 'rams'
}

try:
    # Connect to the MySQL database
    connection = mysql.connector.connect(**db_config)

    # Create a cursor object to interact with the database
    cursor = connection.cursor()

    # Define the department you want to retrieve employees for
    target_department = 'IT'

    # Retrieve employees in the specified department
    cursor.execute("SELECT name FROM employees WHERE department = %s", (target_department,))
    employees = cursor.fetchall()

    # Print the list of employees in the department
    print(f"Employees in the {target_department} department:")
    for employee in employees:
        print(employee[0])

    # Define the employee whose salary you want to update and the new salary
    employee_name_to_update = 'John Doe'
    new_salary = 65000.00

    # Update the salary of the specified employee
    cursor.execute("UPDATE employees SET salary = %s WHERE name = %s", (new_salary, employee_name_to_update))

    # Commit the changes to the database
    connection.commit()

    print(f"Salary of {employee_name_to_update} updated successfully.")
except mysql.connector.Error as err:
    print(f"Error: {err}")
finally:
    if connection.is_connected():
        cursor.close()
        connection.close()


# In[ ]:




