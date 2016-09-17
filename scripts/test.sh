#!/usr/bin/env bash

# Test with HTML coverage
nosetests -w tests --with-coverage --cover-html --cover-package=src
# Test with XML coverage (Cobertura compliant)
nosetests -w tests --with-coverage --cover-xml --cover-package=src
# Test with XUnit xml report
nosetests -w tests --with-xunit