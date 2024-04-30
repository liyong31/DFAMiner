import os
import re
import sys

def list_files(directory):
    # Check if the directory exists
    if not os.path.isdir(directory):
        print("Directory does not exist.")
        return []

    # Get a list of all files in the directory
    files = os.listdir(directory)

    # Print the list of file names
    print("Files in", directory, ":")
    files_lst = []
    for file in files:
        #print(file)
        files_lst.append(file)
    
    return files_lst

def partition_string(main_string, second_string):
    # Find the index where the second string starts
    index = main_string.find(second_string)

    pad_str = ""
    if index == -1:
        # no res file
        pattern = r'\d+-'
        match_str = re.search(pattern, main_string)
        if match_str:
            match_str = match_str.group()
            pad_str = "res"
            print(match_str, pad_str)
            second_string = str(match_str)
        else: 
            return None, None, None  # If the second string is not found, return None for all parts

    # Perform partitioning
    before, separator, after = main_string.partition(second_string)
    # print(before)
    # print(separator)
    # print(after)
    return before, pad_str + separator + after

def extract_and_convert_to_double(file_path, search_string):
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if search_string in line:
                    # Find the position of the search string
                    index = line.find(search_string)
                    if index != -1:
                        # Extract the substring after the search string
                        match = re.search(r"[-+]?\d*\.\d+|\d+", line)
                        match = match.group()
                        print("number: ", match)
                        try:
                            # Convert the substring to a double number (float)
                            double_number = float(match)
                            print("number: ", double_number)
                            return double_number
                            #print("Found '{}' with value: {}".format(search_string, double_number))
                        except ValueError:
                            #print("Found '{}', but the value '{}' is not a valid double number.".format(search_string, substring))
                            print("End of file reached.")
    except FileNotFoundError:
        print("File not found:", file_path)
    except Exception as e:
        print("An error occurred:", e)


# Example usage

path="./dataset"
tools = ["dfaind", "dfa2ind", "dfaID", "dfamin", "sdfamin"]
match_strings = ["Whole tasktime:", "Elapsed time in dfa-identify:", "Elapsed time in miner:"]
solved_nums = [0, 0, 0, 0, 0]
tree_size_strings = ["The APTA size:", "Input SDFA size:"]
separator = "res"
# Iterate over numbers from 1 to 10
max_num = 1200
all_results = []
for i in range(4, 17):
    print(i)
    dir_path = path + "/" + str(i)
    files_lst = list_files(dir_path)

    results = {}
    for file_str in files_lst:
        print("partition :" , file_str)
        tool, file = partition_string(file_str, separator)
        if tool == "":
            continue
        print(tool, "..")
        print(file, ".")
        number = None
        if tool == tools[0] or tool == tools[1]:
            number = extract_and_convert_to_double(dir_path + "/" + file_str, match_strings[0])
        elif tool == tools[2]:
            number = extract_and_convert_to_double(dir_path + "/" + file_str, match_strings[1])
        else:
            number = extract_and_convert_to_double(dir_path + "/" + file_str, match_strings[2])
        if number is None:
            number = max_num
        else:
            for k in range(0, len(tools)):
                solved_nums[k] = solved_nums[k] + 1
            print(number)
        if file in results:
            value = results[file]
            value.add((tool, number))
        else:
            value = set()
            value.add((tool, number))
            results[file] = value
    # print(results)
    all_results.append(results)
#sys.exit(0)
print(all_results)
# now we make sure that
data = [[] for i in range(0, len(tools))]

for d in all_results:
    for i, (k, v) in enumerate(d.items()):
        # k is file name, v is a set of tuples
        for j in range(0, len(tools)):
            # print(j, tools[j])
            # print(j)
            # print(v)
            for tool, time in v:
                # print(tool, time)
                if tool == tools[j]:
                    data[j].append(time)

print(len(all_results))


margin = 80

import matplotlib.pyplot as plt

def get_scatter_plot(x, y, x_tool, y_tool, file, upper):
    # Create scatter plot
    plt.figure(figsize=(4,3), dpi=300)
    plt.scatter(x, y)

    # Set bounds for x and y axes
    plt.xlim(0, max_num)  # Setting x-axis limits from 0 to 6
    plt.ylim(0, max_num) # Setting y-axis limits from 0 to 12

    plt.plot([0, max_num], [0, max_num], color='black', linestyle='--')  # Diagonal line from (0,0) to (12,12)
    plt.plot([0, max_num], [0, max_num/3], color='grey', linestyle='--')  # Diagonal line from (0,0) to (12,12)
    plt.text(max_num - margin, max_num/3 - margin, "3x")

    plt.plot([0, max_num], [0, max_num / 2], color='grey', linestyle='--')  # Diagonal line from (0,0) to (12,12)
    plt.text(max_num - margin, max_num/2-margin, "2x")

    plt.plot([0, max_num], [0, max_num / 4], color='grey', linestyle='--')  # Diagonal line from (0,0) to (12,12)
    plt.text(max_num - margin, max_num / 4 - margin, "4x")

    if upper:

        plt.plot([0, max_num / 3], [0, max_num], color='grey', linestyle='--')  # Diagonal line from (0,0) to (12,12)
        plt.text(max_num/3 - margin, max_num - margin, "3x")

        plt.plot([0, max_num / 2], [0, max_num], color='grey', linestyle='--')  # Diagonal line from (0,0) to (12,12)
        plt.text(max_num/2 - margin, max_num - margin, "2x")

        plt.plot([0, max_num / 4], [0, max_num], color='grey', linestyle='--')  # Diagonal line from (0,0) to (12,12)
        plt.text(max_num / 4 - margin, max_num - margin, "4x")



    # Add labels and title
    plt.xlabel(x_tool)
    plt.ylabel(y_tool)
    #plt.title('Scatter plot on runtime (secs)')

    # Show plot
    #plt.show()
    plt.savefig(file, bbox_inches="tight")
    plt.clf()
    plt.cla()
    plt.close()

tool_names = ['DFA-Inductor','DFA-Inductor2', 'DFA-Identify' , 'DFAMiner-dDFA', 'DFAMiner-3DFA']
# print(data[1])
# print(data[2])
print(solved_nums)
# Sample data
get_scatter_plot(data[0], data[3], tool_names[0], tool_names[3], "ind-3nfa.pdf", False)
get_scatter_plot(data[0], data[4], tool_names[0], tool_names[4], "ind-3dfa.pdf", False)
get_scatter_plot(data[3], data[4], tool_names[3], tool_names[4], "nfa-dfa.pdf", True)
# print(len(data[1]), len(data[3]))
get_scatter_plot(data[1], data[3], tool_names[1], tool_names[3], "ind2-3nfa.pdf", True)
get_scatter_plot(data[1], data[4], tool_names[1], tool_names[4], "ind2-3dfa.pdf", True)
get_scatter_plot(data[2], data[3], tool_names[2], tool_names[3], "id-3nfa.pdf", False)
get_scatter_plot(data[2], data[4], tool_names[2], tool_names[4], "id-3dfa.pdf", False)

for d in all_results:
    for i, (k, v) in enumerate(d.items()):
        # k is file name, v is a set of tuples
        for j in range(0, len(tools)):
            # print(j, tools[j])
            # print(j)
            # print(v)
            solved = {}
            for tool in tools:
                solved[tool] = False
            for tool, time in v:
                # print(tool, time)
                if time < max_num:
                    solved[tool] = True
            if not solved[tools[4]]   and solved[tools[1]]:
                print("Suprise: ", k)
                continue





        


