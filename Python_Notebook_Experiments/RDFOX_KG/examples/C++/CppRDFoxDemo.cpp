// Copyright 2021-2023 by Oxford Semantic Technologies Limited.

#include <algorithm>
#include <iostream>
#include <cstring>

#include <CRDFox/CRDFox.h>

static size_t getTriplesCount(CDataStoreConnection& dataStoreConnection, const char* queryDomain) {
    CParametersPtr parameters = CParameters::newEmptyParameters();
    parameters->setString("fact-domain", queryDomain);

    CCursorPtr cursor = dataStoreConnection.createCursor("SELECT ?X ?Y ?Z WHERE { ?X ?Y ?Z }", 34, *parameters);
    dataStoreConnection.beginTransaction(CTransactionType::READ_ONLY);
    size_t result = 0;
    for (size_t multiplicity = cursor->open(0); multiplicity != 0; multiplicity = cursor->advance())
        result += multiplicity;
    dataStoreConnection.rollbackTransaction();
    return result;
}

bool stdoutOutputStreamFlush(void* context) {
    return fflush(stdout) == 0;
}

bool stdoutOutputStreamWrite(void* context, const void* data, size_t numberOfBytesToWrite) {
    return fwrite(data, sizeof(char), numberOfBytesToWrite, stdout) == numberOfBytesToWrite;
}

struct MemoryInputStreamContext {

public:

    const char* const m_begin;
    const char* const m_afterEnd;
    const char* m_current;

    MemoryInputStreamContext(const char* const begin, const size_t size) :
        m_begin(begin),
        m_afterEnd(m_begin + size),
        m_current(m_begin)
    {
    }

};

bool memoryInputStreamRewind(void* context) {
    MemoryInputStreamContext& memoryInputStreamContext = *reinterpret_cast<MemoryInputStreamContext*>(context);
    memoryInputStreamContext.m_current = memoryInputStreamContext.m_begin;
    return false;
}

bool memoryInputStreamRead(void* context, void* data, size_t numberOfBytesToRead, size_t* bytesRead) {
    MemoryInputStreamContext& memoryInputStreamContext = *reinterpret_cast<MemoryInputStreamContext*>(context);
    const size_t bytesRemaining = static_cast<size_t>(memoryInputStreamContext.m_afterEnd - memoryInputStreamContext.m_current);
    numberOfBytesToRead = std::min<size_t>(numberOfBytesToRead, bytesRemaining);
    std::memcpy(data, memoryInputStreamContext.m_current, numberOfBytesToRead);
    *bytesRead = numberOfBytesToRead;
    memoryInputStreamContext.m_current += *bytesRead;
    return memoryInputStreamContext.m_current == memoryInputStreamContext.m_afterEnd;
}

int main() {
    CRDFoxServer::start(CParameters::getEmptyParameters());

    CRDFoxServer::createFirstRole("guest", "guest");

    CServerConnectionPtr serverConnection = CRDFoxServer::newServerConnection("guest", "guest");

    // We next specify how many threads the server should use during import of data and reasoning.
    std::cout << "Setting the number of threads..." << std::endl;
    serverConnection->setNumberOfThreads(2);

    // We the default value for the "type" perameter, which is "parallel-nn".
    serverConnection->createDataStore("example", CParameters::getEmptyParameters());

    // We connect to the data store.
    CDataStoreConnectionPtr dataStoreConnection = serverConnection->newDataStoreConnection("example");

    // We next import the RDF data into the store. At present, only Turtle/N-triples files are supported.
    // At the moment, please convert RDF/XML files into Turtle format to load into CRDFox.
    std::cout << "Importing RDF data..." << std::endl;

    // To show how to handle exceptions, try to import a file that does not exist.
    std::cout << "To show how to handle exceptions, try to import a file that does not exist." << std::endl;
    try {
        dataStoreConnection->importDataFromFile(nullptr, CUpdateType::ADDITION, "no_file.ttl", "text/turtle");
    }
    catch (const CRDFoxException& exception) {
        std::cout << "Exception:" << std::endl;
        std::cout << "Name: " << exception.getExceptionName() << std::endl;
        std::cout << "What: " <<  exception.what() << std::endl;
    }

    dataStoreConnection->importDataFromFile(nullptr, CUpdateType::ADDITION, "lubm1.ttl", "text/turtle");

    // RDFox manages data in several fact domains.
    //
    // - The 'all' domain contains all facts -- that is, both the explicitly given and the derived facts.
    //
    // - The 'derived' domain contains the facts that were derived by reasoning, but were not explicitly given in the input.
    //
    // - The 'explicit' domain contains the facts that were explicitly given in the input.
    //
    // The domain must be specified in various places where queries are evaluated. If a query domain is not
    // specified, the 'all' domain is used.
    std::cout << "Number of tuples after import: " << getTriplesCount(*dataStoreConnection, "all") << std::endl;

    // SPARQL queries can be evaluated in several ways. One option is to have the query result be written to
    // an output stream in one of the supported formats.
    COutputStream outputStream{ nullptr, &stdoutOutputStreamFlush, &stdoutOutputStreamWrite };
    CStatementResult statementResult = dataStoreConnection->evaluateStatement("SELECT DISTINCT ?Y WHERE { ?X ?Y ?Z }", 37, CParameters::getEmptyParameters(), outputStream, "application/sparql-results+json");
    std::cout << "Query produced " << statementResult.numberOfQueryAnswers << " answers." << std::endl;

    // We now add the ontology and the custom rules to the data.

    // In this example, the rules are kept in a file separate from the ontology. JRDFox supports
    // SWRL rules, so it is possible to store the rules into the OWL ontology.

    std::cout << "Adding the ontology to the store..." << std::endl;
    dataStoreConnection->importDataFromFile(nullptr, CUpdateType::ADDITION, "univ-bench.owl", "text/owl-functional");

    std::cout << "Importing rules from a file..." << std::endl;
    dataStoreConnection->importDataFromFile(nullptr, CUpdateType::ADDITION, "additional-rules.txt", "");

    std::cout << "Number of tuples after materialization: " << getTriplesCount(*dataStoreConnection, "all") << std::endl;

    // We now evaluate the same query as before, but we do so using a cursor, which provides us with
    // programmatic access to individual query results.

    CCursorPtr cursor = dataStoreConnection->createCursor("SELECT DISTINCT ?Y WHERE { ?X ?Y ?Z }", 37, CParameters::getEmptyParameters());

    int numberOfRows = 0;
    std::cout << "\n=======================================================================================" << std::endl;

    size_t arity = cursor->getArity();
    for (size_t termIndex = 0; termIndex < arity; ++termIndex) {
        if (termIndex != 0)
            std::cout << "  ";
        // For each variable, we print the name.
        std::cout << '?' << cursor->getAnswerVariableName(termIndex);
    }
    std::cout << "\n---------------------------------------------------------------------------------------" << std::endl;

    // We iterate through the result tuples.
    for (size_t multiplicity = cursor->open(0); multiplicity != 0; multiplicity = cursor->advance()) {
        ++numberOfRows;
        // We iterate through the terms of each tuple.
        for (size_t termIndex = 0; termIndex < arity; ++termIndex) {
            if (termIndex != 0)
                std::cout << "  ";
            // For each term, we retrieve the lexical form and the data type of the term.
            CDatatypeID datatypeID;
            char lexicalFormBuffer[1024];
            size_t lexicalFormSize = 0;
            cursor->appendResourceLexicalForm(termIndex, lexicalFormBuffer, sizeof(lexicalFormBuffer), lexicalFormSize, datatypeID);
            if (lexicalFormSize >= sizeof(lexicalFormBuffer))
                std::cout << "Warning: the lexical form for the term bound to variable ?" << cursor->getAnswerVariableName(termIndex) << " on row " << numberOfRows << " was truncated from " << lexicalFormSize << " to " << sizeof(lexicalFormBuffer) - 1 << " characters." << std::endl;
            std::cout << lexicalFormBuffer;
        }
        std::cout << std::endl;
    }
    std::cout << "---------------------------------------------------------------------------------------" << std::endl;
    std::cout << "  The number of rows returned: " << numberOfRows << std::endl;
    std::cout << "=======================================================================================\n" << std::endl;

    // RDFox supports incremental reasoning. One can import facts into the store incrementally by
    // calling importDataFromFile(dataStoreConnection, ...) with additional argument CUpdateType::ADDITION.
    std::cout << "Importing triples for incremental reasoning..." << std::endl;
    dataStoreConnection->importDataFromFile(nullptr, CUpdateType::ADDITION, "lubm1-new.ttl", "text/turtle");

    // Adding the rules/facts changes the number of triples. Note that the store is updated incrementally.
    std::cout << "Number of tuples after addition: " << getTriplesCount(*dataStoreConnection, "all") << '\n' << std::endl;

    // Content can be imported from buffers or from streams as well as from file. Here we will add one triple stored
    // in a c string using CDataStoreConnection::importDataFromBuffer and then remove the same content using an in-memory
    // implementation of CInputStream for the same c string. Each time we will count the triples for comparison.
    std::cout << "Adding hard-coded triple..." << std::endl;
    const char* hardCodedTriple = "<http://example.com/subject> <http://example.com/predicate> <http://example.com/object> .";
    const size_t hardCodedTripleLength = std::strlen(hardCodedTriple);
    dataStoreConnection->importDataFromBuffer(nullptr, CUpdateType::ADDITION, hardCodedTriple, hardCodedTripleLength, "text/turtle");
    std::cout << "Number of tuples after adding hard-coded triple: " << getTriplesCount(*dataStoreConnection, "all") << '\n' << std::endl;
    std::cout << "Removing hard-coded triple..." << std::endl;
    MemoryInputStreamContext memoryInputStreamContext(hardCodedTriple, hardCodedTripleLength);
    CInputStream inputStream{ &memoryInputStreamContext, &memoryInputStreamRewind, &memoryInputStreamRead };
    dataStoreConnection->importData(nullptr, CUpdateType::ADDITION, inputStream, "https://rdfox.com/default-base-iri/", "text/turtle");
    std::cout << "Number of tuples after removing hard-coded triple: " << getTriplesCount(*dataStoreConnection, "all") << '\n' << std::endl;

    // One can export the facts from the current store into a file as follows.
    std::cout << "Exporting facts to file 'final-facts.ttl'..." << std::endl;
    CParametersPtr parameters = CParameters::newEmptyParameters();
    parameters->setString("fact-domain", "all");
    dataStoreConnection->exportDataToFile("final-facts.ttl", "application/n-triples", *parameters);

    std::cout << "This is the end of the example!" << std::endl;
    return 0;
}
