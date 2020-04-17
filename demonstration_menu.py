"""
A simple program to demonstrate the tasks (e. g. play button).

Started 17.04.2020.
"""

from source.run_iristasks import task1 as iris1, task2 as iris2
from source.run_voweltasks import task1 as vowel1, task2 as vowel2


def iris2_wrapper():
    print("This task produces a histogram, please choose the relative path to save the figure.")
    print("Select \"n\" to display the figure interactively instead of saving.")
    path = input("Figure path: ")
    if path == "n":
        iris2()
    else:
        iris2(path)


function_lookup = {"iris 1": iris1, "iris 2": iris2_wrapper, "vowel 1": vowel1, "vowel 2": vowel2}


def main():
    print("This is the interactive menu for running the iris and vowel classifier task.")
    print("These tasks were done by Stian Hope and Torstein Nordgård-Hansen\n")

    while True:
        print("\nPlease enter a task to run (iris 1, iris 2, vowel 1 or vowel 2).")
        print("Enter \"quit\" to exit the program.")
        command = input("Command: ")

        if command == "quit":
            break
        elif command in function_lookup.keys():
            function_lookup[command]()
        else:
            print("Command not recognized as one of \"quit\", \"iris 1\", \"iris 2\", \"vowel 1\" or \"vowel 2\".")

    print("Goodbye.")


if __name__ == "__main__":
    main()
