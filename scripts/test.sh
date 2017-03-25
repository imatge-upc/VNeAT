#!/bin/bash

# Test with HTML coverage
nosetests -w tests --with-coverage --cover-html --cover-package=vneat
# Test with XML coverage (Cobertura compliant)
nosetests -w tests --with-coverage --cover-xml --cover-package=vneat
# Test with XUnit xml report
nosetests -w tests --with-xunit