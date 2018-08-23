# natural-language-utils
We are sharing this for educational purposes. This directory contains some helpful natural language utils to create your own cloud-based micro AI services, eg. for chatbots. The code has been written in the form of self-contained micro services based on nanomsg and contains code on intent inference, deep learning code, entity recognition (person, location, organization, ...), syntax analysis, similarity recommendations using approximate nearest neighbors, conversational AI ...  

## Cloud-based Nanoapps for testing

We use Elixir to provide api services and for quick testing we are using an Elixir-Python Bridge. Bots/Services written in python can talk to elixir using the distributed messaging library nanomsg (http://nanomsg.org/ ; https://github.com/nanomsg/nanomsg)
Nanomsg is fast and flexible. Your services can run locally or on a remote machine
 * Python has nanomsg wrapper: https://github.com/tonysimpson/nanomsg-python
 * Elixir to Python: https://github.com/walkr/exns 
 * Elixir to C++ :  https://github.com/wisoltech/elixir-cpp 

Nanobots implements these core concepts of bots using Elixir-Python bridge. Elixir is awesome in pattern matching and in distributed architecture. Python is great in machine learning libraries and in its simplicity.
