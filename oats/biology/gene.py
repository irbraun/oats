



class Gene:

	"""A class representing a single gene with information about its species and different identifiers.
	
	Attributes:
	    all_identifiers (list of str): The combined list of all the strings which represent this gene, both uniquely and not uniquely. 
	    gene_models (list of str): Strings that refer specifically to gene model names which map to this gene, not necessarily uniquely.
	    other_identifiers (list of str): Names, aliases, synonyms, gene models that are mapped to this gene but are not necessarily unique to it.
	    primary_identifier (str): The primary identifer for this gene. 
	    species (str): A string referencing what species this particular gene is in.
	    unique_identifiers (list of str): A list of all the strings that uniquely identify this gene, such as names, accessions, or identifiers.
	"""
	
	def __init__(self, species, unique_identifiers, other_identifiers, gene_models, primary_identifier=None):
		"""Summary
		
		Args:
		    species (str): Description
		
		    unique_identifiers (str): Description
		
		    other_identifiers (str): Description
		
		    gene_models (TYPE): Description
		
		    primary_identifier (str): Description
		
		"""

		# Use the first unique identifier as the primary one is a specific one is not provided.
		if primary_identifier is not None:
			self.primary_identifier = primary_identifier
		else:
			self.primary_identifier = unique_identifiers[0]


		# Set the rest of the attributes directly from the arguments.
		self.species = species
		self.unique_identifiers = unique_identifiers
		self.other_identifiers = other_identifiers
		self.gene_models = gene_models
		self.all_identifiers = unique_identifiers+other_identifiers