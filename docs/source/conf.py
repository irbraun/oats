# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import sphinx_bootstrap_theme
sys.path.insert(0, os.path.abspath('../..'))






# Because the environment for building the documentation has to be very ligthweight, it is different
# than the conda environment used for actually running tests or using the methods in the package. 
# However, autodoc still requires that all the modules be genuinely importable, which means that
# they can't be importing packages that are not available or present in this lightweight environment,
# even if none of those methods are actually going to be called while building the documentation.
# The solution is to mock all of those modules that are imported by any of the modules looked at by 
# autodoc.

from mock import Mock as MagicMock
class Mock(MagicMock):
	@classmethod
	def __getattr__(cls, name):
		return MagicMock()
 
MOCK_MODULES = ['collections.defaultdict','pandas','glob']
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)







# -- Project information -----------------------------------------------------

project = 'oats'
copyright = '2020, Ian Braun'
author = 'Ian Braun'

# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
	'sphinx.ext.autodoc',
	'sphinx_bootstrap_theme'
]



# Had to add this too.
master_doc = 'index'



# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'bootstrap'


# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
	# Bootswatch (http://bootswatch.com/) theme.
	"bootswatch_theme": "flatly",
	# Choose Bootstrap version.
	"bootstrap_version": "3",
	# Tab name for entire site. (Default: "Site")
	"navbar_site_name": "Documentation",
	# HTML navbar class (Default: "navbar") to attach to <div> element.
	# For black navbar, do "navbar navbar-inverse"
	"navbar_class": "navbar",
	# Render the next and previous page links in navbar. (Default: true)
	"navbar_sidebarrel": True,
	# Render the current pages TOC in the navbar. (Default: true)
	"navbar_pagenav": False,
	# A list of tuples containing pages or urls to link to.
	#"navbar_links": [
	#    ("GitHub", _parser.get("metadata", "home-page").strip(), True)
	#] + [
	#    (k, v, True)
	#    for k, v in project_urls.items()
	#    if k not in {"Documentation", "Changelog"}
	#],
	"admonition_use_panel": True,
}

html_sidebars = {
	"*": ["localtoc.html"],
	"examples/*": ["localtoc.html"],
	"api/*": ["localtoc.html"],
}






# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
