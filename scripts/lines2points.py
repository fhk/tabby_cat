
import sys
import csv

from shapely.wkt import loads
from pyproj import Proj, transform



class Line2Point():

    customer = 1
    main_run = 2
    customers_u = set()
    inProj = Proj(init='epsg:2163')
    outProj = Proj(init='epsg:4326')
    output = ['type', 'length', 'lat', 'lon']

    def __init__(self, lines):
        self.lines = lines

    def parse_customers(self):
        for row in self.customers:
            #lat, lon
            self.customers_u.add((float(row[0]), float(row[1])))
        print(len(self.customers_u))

    def parse_network(self):

        with open('hex_input.csv', 'w') as hex_i:
            writer = csv.writer(hex_i)
            writer.writerow(self.output)
            for row in self.lines:
                geom = loads(row[0])
                x1, y1 = geom.centroid.coords[0]
                x2,y2 = transform(self.inProj,self.outProj,x1,y1)

                # if transform(self.inProj,self.outProj, geom.coords[0][0], geom.coords[0][1]) in self.customers_u or transform(self.inProj,self.outProj, geom.coords[-1][0], geom.coords[-1][1]) in self.customers_u:
                #     writer.writerow([self.customer, geom.length, y2, x2])
                # else:
                writer.writerow([self.main_run, geom.length, y2, x2])


def main():

    #customer = sys.argv[2]
    network = sys.argv[1]

    #with open(customer, 'r') as c_data:
    with open(network, 'r') as n_data:
        #c_csv = csv.reader(c_data)
        #next(c_csv)
        n_csv = csv.reader(n_data)
        lp = Line2Point(n_csv)
        #lp.parse_customers()
        lp.parse_network()



if __name__ == "__main__":
    main()

    