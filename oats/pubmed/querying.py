from Bio import Entrez

def search(query,limit):
	Entrez.email = 'a@b.com'
	handle = Entrez.esearch(db='pubmed',sort='relevance',retmax=limit,retmode='xml',term=query)
	results = Entrez.read(handle)
	return results


def fetch_details(id_list):
	ids = ','.join(id_list)
	Entrez.email = 'a@b.com'
	handle = Entrez.efetch(db='pubmed', retmode='xml', id=ids)
	results = Entrez.read(handle)
	return results