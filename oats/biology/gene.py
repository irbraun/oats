



class Gene:

	"""A class representing a single gene
	
	Attributes:
	    names (list of str): A list of all the strings that identify this gene, such as names, accessions, or identifiers.
	    species (str): A string referencing what species this particular gene is in.
	"""
	
	def __init__(self, names, species):
		self.names = names
		self.species = species