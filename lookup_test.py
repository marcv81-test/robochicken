# TODO: write proper unit tests

from lookup import *

from pprint import pprint

lookup_table = LookupTable(1, 1, 3)

pprint(lookup_table.input_vector_to_index([0.0]))
pprint(lookup_table.input_vector_to_index([0.5]))
pprint(lookup_table.input_vector_to_index([1.0]))

lookup_table = LookupTable(1, 1, 6)

pprint(lookup_table.input_vector_to_index([0.0]))
pprint(lookup_table.input_vector_to_index([0.2]))
pprint(lookup_table.input_vector_to_index([0.4]))
pprint(lookup_table.input_vector_to_index([0.6]))
pprint(lookup_table.input_vector_to_index([0.8]))
pprint(lookup_table.input_vector_to_index([1.0]))

lookup_table = LookupTable(2, 1, 3)

pprint(lookup_table.input_vector_to_index([0.0, 0.0]))
pprint(lookup_table.input_vector_to_index([0.0, 0.5]))
pprint(lookup_table.input_vector_to_index([0.0, 1.0]))
pprint(lookup_table.input_vector_to_index([0.5, 0.0]))
pprint(lookup_table.input_vector_to_index([0.5, 0.5]))
pprint(lookup_table.input_vector_to_index([0.5, 1.0]))
pprint(lookup_table.input_vector_to_index([1.0, 0.0]))
pprint(lookup_table.input_vector_to_index([1.0, 0.5]))
pprint(lookup_table.input_vector_to_index([1.0, 1.0]))

lookup_table = LookupTable(4, 4, 6)
pprint(lookup_table.input_vector_to_index([1.0, 1.0, 1.0, 1.0])) #(6**4)-1
pprint(lookup_table.input_vector_to_index([0.6, 0.8, 0.2, 1.0])) #(6**4)-1

pprint(lookup_table.index_to_input_vector(803))

lookup_table.populate(lambda x: x)
pprint(lookup_table._table)
