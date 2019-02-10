#!/usr/bin/python

# Copyright (C) 2018, Arun Thundyill Saseendran {ats0stv@gmail.com, thundyia@tcd.ie},
# Viren Chhabria {chhabriv@tcd.ie, viren.chhabria@gmail.com},
# Debrup Chakraborty {chakrabd@tcd.ie, rupdeb@gmail.com},
# Lovish Setia {setial@tcd.ie}, Aneek Barman Roy {barmanra@tcd.ie}
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import random
import sys
import os
import pandas as pd
import numpy as np
import math


# find the datatype of the column
def find_datatype(value):
    try:
        int(value)
        return 'integer'
    except ValueError:
        try:
            float(value)
            return 'float'
        except ValueError:
            return 'string'

# find the maximum and minimum for ech column


def get_min_max(str_path):

    global list_minimum
    global list_maximum
    os.chdir(os.path.dirname(str_path))

    data = pd.read_csv(os.path.basename(str_path))

    for i in data.columns:
        list_column = np.array(data[i])

        # each member is a type of np and not integer or string
        list_maximum.append(np.ndarray.max(list_column))
        list_minimum.append(np.ndarray.min(list_column))

    # Added the 5 percent +- range to  the noise
    for i in range(len(list_maximum)):
        #print('value:{} and type:{}'.format(list_maximum[i], type(list_maximum[i])))
        if isinstance(list_minimum[i], str):
            continue

        if isinstance(list_maximum[i], np.int64):
            list_maximum[i] = math.ceil(
                list_maximum[i] + (list_maximum[i] * 0.05))
        elif isinstance(list_maximum[i], np.float64):
            list_maximum[i] = list_maximum[i] + (list_maximum[i] * 0.05)

        if isinstance(list_minimum[i], np.int64):
            list_minimum[i] = math.floor(
                list_minimum[i] - (list_minimum[i] * 0.05))
        elif isinstance(list_minimum[i], np.float64):
            list_minimum[i] = list_minimum[i] - (list_minimum[i] * 0.05)

# Add noisy data to the existing file


def add_noisy_data(noise_perentage, path_of_file, path_of_new_file):

    noise_list = []
    counter = 0
    type_list = []
    list_members = []
    try:
        lines = [line.rstrip('\n') for line in open(path_of_file, 'r')]
    except IOError:
        print(
            'An error has been encountered while getting all the information from the file')

    # Choose 16th column to predict the type of columns
    with open(path_of_file, 'r') as dataset:
        for line in dataset:
            counter += 1
            if counter == 16:
                list_members = line.split(',')
                break

    # Number of columns that will be added as noise
    noise_count = int(len(lines)*(int(noise_perentage)/100))

    # Find the different kind of data-types available in the dataset
    for member in list_members:  # each member is a string
        type_list.append(find_datatype(member))
    # Create noise based on the data type
    for j in range(noise_count):
        list_random = []
        for i in range(len(list_members)):
            if type_list[i] == 'string':
                list_random.append(list_members[i])
            elif type_list[i] == 'float':
                list_random.append(
                    str(round(random.uniform(list_minimum[i], list_maximum[i]), 2)))
            elif type_list[i] == 'integer':
                list_random.append(
                    str(random.randint(int(list_minimum[i]), int(list_maximum[i]))))

        noise_list.append((','.join(list_random)))

    # Adding an extra column to differ. original data and noise
    lines = [line + ',0' for line in lines]
    noise_list = [line + ',1' for line in noise_list]

    # Combine the actual data and noise and shuffle the list
    lines += noise_list
    copy_lines = lines[1:]
    random.shuffle(copy_lines)
    lines[1:] = copy_lines
    lines[0] = lines[0][:len(lines[0]) - 1]
    lines[0] = lines[0] + 'Noise'

    # Write the new dataset to the file
    try:
        print('creating a file:{}'.format(path_of_new_file))
        with open(path_of_new_file, 'w') as dataset_copy:
            for line in lines:
                print(line, file=dataset_copy)
    except IOError:
        print('The new file path mentioned is in use please close the file and retry again')


def add_feature_based_on_result(result_feature, no_of_feature, path_of_new_file):

    lines = [line.rstrip('\n') for line in open(path_of_file, 'r')]
    result_sub = 0
    list_line = lines[0].split(',')
    result_sub = list_line.index(result_feature)

    for i in range(1, len(lines)):
        list_line = lines[i].split(',')

        list_line.append(str(random.randint(0, 100)))

        if no_of_feature > 1:
            list_line.append(str(float(list_line[result_sub])**3))

        if no_of_feature > 2:
            list_line.append(
                (str(1/(1+math.sin(float(list_line[result_sub+1]))))))

        if no_of_feature > 3:
            list_line.append(
                str(abs(math.cos(float(list_line[result_sub])**3))))

        if no_of_feature > 4:
            list_line.append(str(random.randint(0, 500)))

        lines[i] = (',').join(list_line)

    for i in range(no_of_feature):
        lines[0] += ',feature_y_'+str(no_of_feature)

    print('creating a file:{}'.format(path_of_new_file))
    with open(path_of_new_file, 'w') as write_file:
        for i in range(len(lines)):
            print(lines[i], file=write_file)


def add_feature_value(str_line, int_feature):

    list_line = str_line.split(',')

    if int_feature == 1:
        return (str(random.randint(0, 100)))
    elif int_feature == 2:
        return ((str(random.randint(500, 1000))))
    elif int_feature == 3:
        return (str(random.randint(0, 250)))
    elif int_feature == 4:
        return (str(random.randint(500, 750)))
    else:
        return(str(str(random.randint(0, 500))))


def add_noise_attributes(feature_count, path_of_file, path_of_new_file):
    counter = 0

    # Read the dataset without newline character
    lines = [line.rstrip('\n') for line in open(path_of_file, 'r')]

    # Add the new features to every line
    for i in range(len(lines)):
        # Generate a random list of numbers
        list_append = []
        for feature in range(1, int(feature_count)+1):
            # Add the feature name
            if counter == 0:
                list_append.append('feature'+str(feature))
                if feature == int(feature_count):
                    counter = 1
            else:
                # list_append.append(str(random.randint(0,500)))
                list_append.append(add_feature_value(lines[i], feature))

        # Append the random string to every line
        lines[i] += ','
        lines[i] += (',').join(list_append)

    print('creating a file:{}'.format(path_of_new_file))
    # Write to a new file
    try:
        with open(path_of_new_file, 'w') as dataset:
            for line in lines:
                print(line, file=dataset)

    except IOError:
        print('The new file path mentioned is in use please close the file and retry again')


if __name__ == "__main__":

    list_minimum = []
    list_maximum = []

    # remove this code
    #sys.argv = ['1','C:\\Users\\lovishsetia\\Desktop\\Academic\\ML\\Group\\wine_dataset\\dataset_wine.csv', 'C:\\Users\\lovishsetia\\Desktop\\Academic\\ML\\Group\\wine_dataset\\noisy_data1.csv']
    sys.argv = ['0', 'U:\\noisy_data\\datasets\\dataset_bike\\hour_removed_features.csv',
                'U:\\noisy_data\\noisy_data_']
    # remove this code
    try:
        args = sys.argv

        noise_type = args[0].replace("\\", "\\\\")
        path_of_file = args[1]

        get_min_max(path_of_file)

        for i in range(10, 101, 10):
            for j in range(2):
                path_of_new_file = args[2] + str(i)+'_'+str(j+1)+'.csv'
                add_noisy_data(i, path_of_file, path_of_new_file)

        for i in range(1, 6):
            for j in range(1, 3):
                path_of_new_file = args[2] + \
                    'feature_' + str(i)+'_'+str(j)+'.csv'
                add_noise_attributes(i, path_of_file, path_of_new_file)

        # for i in range(1,6):
         #   for j in range(1,3):
          #      path_of_new_file =  args[2] + 'feature_y_'+ str(i)+'_'+str(j)+'.csv'
            #     add_feature_based_on_result('cnt',i,path_of_new_file)

    except IndexError:
        print('Please provide 3 parameters in order of type_of_noise, path_data_set, path_of_new_dataset')
