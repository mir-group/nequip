# Contributing

## Design Philosophy

 - Prioritize extensibility and configurability in core abstractions, we can also make simpler user-interfaces when needed.
 - It is preferable to add features with more code (e.g. subclassing core abstractions) than with more options (e.g. by adding new arguments to core abstractions) wherever possible.
 
 
## Code Standards

 ### Unittests
 
 For new features, write a unittest that covers it wherever possible. If it is a significant change to the training workflow, updating the integrations tests might be required.
 
 ### Documentation
 
  - Add comments for code whose purpose is not immediately obvious.
 
  - For new classes or functions that will be exposed to users, comprehensive user-facing docstrings are a must. We follow Google-style Python docstrings.
  
  - For new classes or functions that are not user-facing, docstrings and explanatory comments are strongly encouraged (and may be required)

 ### Style Enforcement
 All contributions must conform to `flake8` and have the `black` code formatter be run on it.
 
 ### Git Practices
 It is preferable to `rebase` wherever possible, and `merge` only if the situation calls for it. We strive towards a clean and easily-readable commit history.