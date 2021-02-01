import csv

with open('File/TB_data_dictionary_2021-01-28.csv', 'r') as csv_file:
    csv_reader = csv.DictReader(csv_file)


    with open('File/new_csv.csv','w') as new_file:
        fieldnames = ['variable_name', 'code_list', 'definition']

        csv_writer = csv.DictWriter(new_file, fieldnames=fieldnames, delimiter='\t')

        csv_writer.writeheader()

        for line in csv_reader:
            del line['dataset']
            csv_writer.writerow(line)