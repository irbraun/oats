


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
	"EXP":"exp",	# Experimental evidence codes.
	"IDA":"exp",
	"IPI":"exp",
	"IMP":"exp",
	"IGI":"exp",
	"IEP":"exp",
	"HTP":"exp",
	"HDA":"exp",
	"HMP":"exp",
	"HGI":"exp",
	"HEP":"exp",
	"IBA":"phy", 	# Phylogenetically inferred evidence codes.
	"IBD":"phy",
	"IKR":"phy",
	"IRD":"phy",
	"ISS":"comp", 	# Computationally inferred evidence codes.
	"ISO":"comp",
	"ISA":"comp",
	"ISM":"comp",
	"IGC":"comp",
	"RCA":"comp",
	"TAS":"auth",	# Author statement inferred evidence codes.
	"NAS":"auth",
	"IC":"cur",		# Curator statement evidence codes. 
	"ND":"cur",
	"IEA":"elc"		# Electronic annotation evidence codes.
}