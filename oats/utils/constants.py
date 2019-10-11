


# Protected gene prefixes that shouldn't appear anywhere in any gene names.
REFGEN_V3_TAG = "refgen_v3="
REFGEN_V4_TAG = "refgen_v3="
NCBI_TAG = "ncbi="
UNIPROT_TAG = "uniprot="




# Mapping between full names and their abbreviations/codes.
ABBREVIATIONS_MAP = {
	"Zea mays ssp mays":"zma",
	"Arabidopsis thaliana":"ath",
	"Oryza sativa":"osa",
	"Medicago truncatula":"mtr",
	"Glycine max":"gmx",
	"Solanum lycopersicum":"sly"
}





EVIDENCE_CODES = {
	"EXP":"experimental",			# Experimental evidence codes.
	"IDA":"experimental",
	"IPI":"experimental",
	"IMP":"experimental",
	"IGI":"experimental",
	"IEP":"experimental",
	"HTP":"high_throughput",
	"HDA":"high_throughput",
	"HMP":"high_throughput",
	"HGI":"high_throughput",
	"HEP":"high_throughput",
	"IBA":"phylogenetics", 			# Phylogenetically inferred evidence codes.
	"IBD":"phylogenetics",
	"IKR":"phylogenetics",
	"IRD":"phylogenetics",
	"ISS":"computational", 			# Computationally inferred evidence codes.
	"ISO":"computational",
	"ISA":"computational",
	"ISM":"computational",
	"IGC":"computational",
	"RCA":"computational",
	"TAS":"author_statement",		# Author statement inferred evidence codes.
	"NAS":"author_statement",
	"IC":"curator_statement",		# Curator statement evidence codes. 
	"ND":"curator_statement",
	"IEA":"electronic_annotation"	# Electronic annotation evidence codes.
}