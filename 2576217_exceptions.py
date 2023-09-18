#!/usr/bin/env python
# coding: utf-8

# # # 1. Write a python program with Exception handling to input marks for five subjects Physics, Chemistry, Biology, Mathematics, and Computer. Calculate the percentage and grade according to the following:
# # 
# # i) Percentage >= 90% : Grade A
# # 
# # ii) Percentage> 80% : Grade B
# # 
# # Percentage> 70%: Grade C
# # 
# # iv) Percentage >= 60%: Grade D
# # 
# # v) Percentage> 40%: Grade E
# # 
# # vi) Percentage 40%: Grade F

# In[1]:


try:
    # Input marks for each subject
    physics = float(input("Enter Physics marks: "))
    chemistry = float(input("Enter Chemistry marks: "))

    biology = float(input("Enter Biology marks: "))
    mathematics = float(input("Enter Mathematics marks: "))
    computer = float(input("Enter Computer marks: "))

    # Calculate the total marks
    total_marks = physics + chemistry + biology + mathematics + computer

    # Calculate the percentage
    percentage = (total_marks / 500) * 100

    # Determine the grade based on the percentage
    if percentage >= 90:
        grade = "A"
    elif percentage >= 80:
        grade = "B"
    elif percentage >= 70:
        grade = "C"
    elif percentage >= 60:
        grade = "D"
    elif percentage > 40:
        grade = "E"
    else:
        grade = "F"

    # Display the percentage and grade
    print(f"Percentage: {percentage:.2f}%")
    print(f"Grade: {grade}")

except ValueError:
    print("Invalid input. Please enter numeric values for marks.")
except Exception as e:
    print(f"An error occurred: {e}")


# # 2.Write a python program with Exception handling to input electricity unit charges and calculate the total electricity bill according to the given condition:
# # 
# # i)For the first 50 units Rs. 0.50/unit
# # 
# # ii)For the next 100 units Rs. 0.75/unit
# # 
# # iii)For the next 100 units Rs. 1.20/unit
# # 
# # iv)For units above 250 Rs. 1.50/unit
# # 
# # v)An additional surcharge of 20% is added to the bill.

# In[2]:


try:
    # Input the electricity unit charges
    units = float(input("Enter the electricity units consumed: "))

    # Check if the units consumed are valid
    if units < 0:
        raise ValueError("Units consumed cannot be negative")

    # Initialize variables to store the total bill and surcharge
    total_bill = 0
    surcharge = 0

    # Calculate the bill based on the given conditions
    if units <= 50:
        total_bill = units * 0.50
    elif units <= 150:
        total_bill = (50 * 0.50) + ((units - 50) * 0.75)
    elif units <= 250:
        total_bill = (50 * 0.50) + (100 * 0.75) + ((units - 150) * 1.20)
    else:
        total_bill = (50 * 0.50) + (100 * 0.75) + (100 * 1.20) + ((units - 250) * 1.50)

    # Calculate the surcharge (20% of the total bill)
    surcharge = 0.20 * total_bill

    # Add surcharge to the total bill
    total_bill += surcharge

    # Display the total electricity bill
    print(f"Total electricity bill: Rs. {total_bill:.2f}")

except ValueError as ve:
    print(f"Error: {ve}")
except Exception as e:
    print(f"An error occurred: {e}")


# # 3.Write a python program with Exception handling to input the week number and print the weekday.

# In[3]:


try:
    # Input the week number
    week_number = int(input("Enter the week number (1-7): "))

    # Check if the week number is valid (between 1 and 7)
    if 1 <= week_number <= 7:
        # Define a list of weekdays
        weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        # Print the corresponding weekday
        print(f"Weekday for week number {week_number} is {weekdays[week_number - 1]}")
    else:
        print("Invalid week number. Please enter a number between 1 and 7.")

except ValueError:
    print("Invalid input. Please enter a valid week number as an integer.")
except Exception as e:
    print(f"An error occurred: {e}")


# # # 4.Write a Python program to implement word count using command line arguments.
# # 
# # i) Create a text document "apple.txt" whch contains text for wordcount.
# # 
# # ii)Create a wordcount program which calls the "apple.txt" document by opening the file. 
# # 
# # iii)If the word is present again in the "aaple.txt",the wordcount is incremented by 1 until all the words are counted in the document.
# # 
# # iv)Close the file.
# # 
# # v) Create a command.py program which imports the wordcount.py program.
# # 
# # vi) Count the number of words using command line arguments.
# # 
# # vii) Print each word and its count.

# In[5]:


def count_words(filename):
    try:
        with open(filename, 'r') as file:
            text = file.read()
            words = text.split()
            word_count = {}
            for word in words:
                word = word.strip('.,!?()[]{}":;')
                word = word.lower()
                if word:
                    word_count[word] = word_count.get(word, 0) + 1
            return word_count
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
        return {}

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python wordcount_and_command.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]
    word_count = count_words(filename)
    for word, count in word_count.items():
        print(f"{word}: {count}")


# # 5. Write a Python program for finding the most frequent words in a text read from a file.
# # 
# # i) Initially open the text file in read mode.
# #
# # ii) Make all the letters in the document into lowercase letters and split the words in each line. I
# # 
# # iii) Get the words in an order.
# # 
# # iv) Sort the words for finding the most frequent words in the file.

# In[4]:


# Open the text file in read mode
with open('in1.txt', 'r') as file:
    text = file.read()

# Convert all letters to lowercase and split words
words = text.lower().split()

# Create a dictionary to store word frequencies
word_count = {}

# Count the frequency of each word
for word in words:
    if word in word_count:
        word_count[word] += 1
    else:
        word_count[word] = 1

# Sort the words by frequency in descending order
sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

# Print the most frequent words
num_most_frequent = 10  # You can change this to show more or fewer words
for word, frequency in sorted_words[:num_most_frequent]:
    print(f'{word}: {frequency}')


# # # 6.File Processing with Command-Line Arguments Scenario: You are developing a command-line utility that processes text files. Users can specify input and output file paths as command line arguments Your program should handle exceptions gracefully.
# # 
# # i)Design a Python program that takes two command line argument: the input file path and the output file path. Ensure that the program checks if both arguments are provided and that the input file exists.
# # 
# # ii)Implement error handling a deal with scenarios such as missing input files, invalid file path or permison sues when writing to the output file.
# # 
# # iii)If an error occurs during file processing display a user friendly, error message and exit the program with a non zero exit code.
# # 
# # iv) Write test cases that cover various scenarios, including providing valid and invalid file paths as command line arguments.
# 

# In[6]:


import sys

def process_file(input_file_path, output_file_path):
    try:
        # Check if both input and output file paths are provided
        if not input_file_path or not output_file_path:
            raise ValueError("Both input and output file paths must be provided.")

        # Check if the input file exists
        try:
            with open(input_file_path, 'r') as input_file:
                # Read the contents from the input file
                file_contents = input_file.read()

            # Process the file contents (You can add your processing logic here)
            # For this example, we'll just copy the contents to the output file

            with open(output_file_path, 'w') as output_file:
                output_file.write(file_contents)

            print(f"File processing completed. Data copied from '{input_file_path}' to '{output_file_path}'.")
        
        except FileNotFoundError:
            raise FileNotFoundError(f"Input file '{input_file_path}' not found.")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)  # Exit with a non-zero exit code in case of an error

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python your_program.py <input_file_path> <output_file_path>")
        sys.exit(1)
    
    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]

    process_file(input_file_path, output_file_path)


# In[ ]:





# In[ ]:





# In[ ]:




