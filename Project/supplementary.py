import pandas as pd
import re
import numpy as np
import os.path


def read_database():
    """
    Reads the database into a pandas dataframe
    @param
    @return the dataframe
    """

    source = "data/wtf"
    mydata = pd.DataFrame()
    idxsum = 0
    year_range = ['98', '99', "00"]
    for year in year_range:
        temp = pd.read_table(source + year + ".txt", header=0, sep='\t', error_bad_lines=False, dtype={'importer': np.str, 'exporter': np.str, 'sitc4': np.str})
        temp = temp.fillna('')
        temp.importer = temp.importer.astype(str)
        temp.year = temp.year.astype(int)
        temp.value = temp.value.astype(int)
        temp.icode = temp.icode.astype(str)
        temp.ecode = temp.ecode.astype(str)
        temp.exporter = temp.exporter.astype(str)
        temp.sitc4 = temp.sitc4.astype(str)
        mydata = pd.concat([mydata, temp])
        idxsum += len(temp)

    assert idxsum == len(mydata), "Houston we have got a problem, some data is lost out there in space"


    product_space_orig = []
    country_space_orig = []
    rca_matrix_orig = {}
    with open("data/proximity9800.txt") as f:
        for line in f:
            data = filter(len, line.split(' '))
            product_space_orig.append(data[0].replace("\"", "").rstrip('\n')) if data[0].replace("\"", "").rstrip('\n') not in product_space_orig else None
            product_space_orig.append(data[1].replace("\"", "").rstrip('\n')) if data[1].replace("\"", "").rstrip('\n') not in product_space_orig else None
    with open("data/RCA.txt") as f:
        for line in f:
            data = filter(len, line.split(' '))
            country_space_orig.append(data[-1].replace("\"", "").rstrip('\n')) if data[-1].replace("\"", "").rstrip('\n') not in country_space_orig else None
            if data[-1].replace("\"", "").rstrip('\n') not in rca_matrix_orig:
                rca_matrix_orig[data[-1].replace("\"", "").rstrip('\n')] = {}
            if data[0].replace("\"", "").rstrip('\n') not in rca_matrix_orig[data[-1].replace("\"", "").rstrip('\n')]:
                rca_matrix_orig[data[-1].replace("\"", "").rstrip('\n')][data[0].replace("\"", "").rstrip('\n')] = {}
            if int(data[2].replace("\"", "").rstrip('\n')) not in rca_matrix_orig[data[-1].replace("\"", "").rstrip('\n')][data[0].replace("\"", "").rstrip('\n')] and 1998 <= int(data[2].replace("\"", "").rstrip('\n')) <= 2000:
                rca_matrix_orig[data[-1].replace("\"", "").rstrip('\n')][data[0].replace("\"", "").rstrip('\n')][int(data[2].replace("\"", "").rstrip('\n'))] = float(data[3].replace("\"", "").rstrip('\n'))
    return mydata, product_space_orig, country_space_orig, rca_matrix_orig


if __name__ == "__main__":
    numerical_year_range = range(1998, 2001)
    mydata, product_space_orig, country_space_orig, rca_matrix_orig = read_database()
    product_space = []
    country_space = []
    for product in mydata.sitc4.unique():
        if len(str(product)) == 4:
            product_space.append(product)
    for country in mydata.icode.unique():
        country_space.append(country)
    for country in mydata.ecode.unique():
        country_space.append(country) if country not in country_space else None
    print "Houston, we got %d products (out of which %d are present in the original)" % (len(product_space), len(product_space_orig))
    print "Houston, we got %d countries (out of which %d are present in the original)" % (len(country_space), len(country_space_orig))
    if not os.path.isfile("data/rca.npz"):
        RCA_matrix = {}
        for country in country_space_orig:
            RCA_matrix[country] = {}
            for product in product_space_orig:
                sumrca = 0
                for year in numerical_year_range:
                    sum_i_c_xci = mydata[(mydata['year'] == year)]
                    sum_c_xci = sum_i_c_xci[(sum_i_c_xci['sitc4'] == product)]
                    sum_i_xci = sum_i_c_xci[(sum_i_c_xci['exporter'] == country)]
                    xci = sum_i_c_xci[(sum_i_c_xci['exporter'] == country) & (sum_i_c_xci['sitc4'] == product)]
                    if not xci.empty:
                        xci = xci['value'].sum()
                        sum_i_xci = sum_i_xci['value'].sum()
                        sum_c_xci = sum_c_xci['value'].sum()
                        sum_i_c_xci = sum_i_c_xci['value'].sum()
                        temp_rca = ((1.0*xci)/(1.0*sum_i_xci))/((1.0*sum_c_xci)/(1.0*sum_i_c_xci))
                        sumrca += temp_rca
                    else:
                        if country in rca_matrix_orig:
                            if product in rca_matrix_orig[country]:
                                if year in rca_matrix_orig[country][product]:
                                    sumrca += rca_matrix_orig[country][product][year]
                            else:
                                sumrca += 0
                        else:
                            sumrca += 0
                RCA_matrix[country][product] = sumrca/(1.0*len(numerical_year_range))
                print sumrca/(1.0*len(numerical_year_range))
        np.savez("data/rca", RCA_matrix=RCA_matrix)
    else:
        f = np.load("data/rca.npz")
        RCA_matrix = f['RCA_matrix']