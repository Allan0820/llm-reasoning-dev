// Copyright 2021-2023 by Oxford Semantic Technologies Limited.

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <inttypes.h>

#include <CRDFox/CRDFox.h>

static size_t getTriplesCount(CDataStoreConnection* dataStoreConnection, const char* queryDomain) {
    CParameters* parameters = NULL;
    CParameters_newEmptyParameters(&parameters);
    CParameters_setString(parameters, "fact-domain", queryDomain);

    CCursor* cursor = NULL;
    CDataStoreConnection_createCursor(dataStoreConnection, "SELECT ?X ?Y ?Z WHERE { ?X ?Y ?Z }", 34, parameters, &cursor);
    CParameters_destroy(parameters);
    CDataStoreConnection_beginTransaction(dataStoreConnection, CTransactionType_READ_ONLY);
    size_t result = 0;
    size_t multiplicity;
    for (CCursor_open(cursor, 0, &multiplicity); multiplicity != 0; CCursor_advance(cursor, &multiplicity))
        result += multiplicity;
    CCursor_destroy(cursor);
    CDataStoreConnection_rollbackTransaction(dataStoreConnection);
    return result;
}

void handleCRDFoxException(bool exception) {
    if (exception) {
        const char* what;
        size_t whatLength;
        bool isRDFoxException;
        const char* exceptionName;
        size_t exceptionNameLength;
        const char* message;
        size_t messageLength;
        CRDFoxServer_getLastException(&what, &whatLength, &isRDFoxException, &exceptionName, &exceptionNameLength, &message, &messageLength);
        printf("Exception:\n");
        printf("Name: %s\n", exceptionName);
        printf("What: %s\n", what);
        exit(1);
    }
}

bool stdoutOutputStreamFlush(void* context) {
    return fflush(stdout) == 0;
}

bool stdoutOutputStreamWrite(void* context, const void* data, size_t numberOfBytesToWrite) {
    return fwrite(data, sizeof(char), numberOfBytesToWrite, stdout) == numberOfBytesToWrite;
}

typedef struct MemoryInputStreamContext {
    const char* const begin;
    const char* const afterEnd;
    const char* current;
} MemoryInputStreamContext;

bool memoryInputStreamRewind(void* context) {
    struct MemoryInputStreamContext* memoryInputStreamContext = (struct MemoryInputStreamContext*)context;
    memoryInputStreamContext->current = memoryInputStreamContext->begin;
    return false;
}

bool memoryInputStreamRead(void* context, void* data, size_t numberOfBytesToRead, size_t* bytesRead) {
    struct MemoryInputStreamContext* memoryInputStreamContext = (struct MemoryInputStreamContext*)context;
    const size_t bytesRemaining = (size_t)(memoryInputStreamContext->afterEnd - memoryInputStreamContext->current);
    numberOfBytesToRead = numberOfBytesToRead <= bytesRemaining ? numberOfBytesToRead : bytesRemaining;
    memcpy(data, memoryInputStreamContext->current, numberOfBytesToRead);
    *bytesRead = numberOfBytesToRead;
    memoryInputStreamContext->current += *bytesRead;
    return memoryInputStreamContext->current == memoryInputStreamContext->afterEnd;
}

int main() {
    handleCRDFoxException(CRDFoxServer_start(CParameters_getEmptyParameters(), NULL));

    handleCRDFoxException(CRDFoxServer_createFirstRole("guest", "guest"));

    CServerConnection* serverConnection = NULL;
    handleCRDFoxException(CRDFoxServer_newServerConnection("guest", "guest", &serverConnection));

    // We next specify how many threads the server should use during import of data and reasoning.
    printf("Setting the number of threads...\n");
    handleCRDFoxException(CServerConnection_setNumberOfThreads(serverConnection, 2));

    // We the default value for the "type" perameter, which is "par-complex-nn".
    handleCRDFoxException(CServerConnection_createDataStore(serverConnection, "example", CParameters_getEmptyParameters()));

    // We connect to the data store.
    CDataStoreConnection* dataStoreConnection = NULL;
    handleCRDFoxException(CServerConnection_newDataStoreConnection(serverConnection, "example", &dataStoreConnection));

    // We next import the RDF data into the store. At present, only Turtle/N-triples files are supported.
    // At the moment, please convert RDF/XML files into Turtle format to load into CRDFox.
    printf("Importing RDF data...\n");

    // To show how to handle errors, try to import a file that does not exist.
    printf("To show how to handle exceptions, try to import a file that does not exist.\n");
    if (CDataStoreConnection_importDataFromFile(dataStoreConnection, NULL, CUpdateType_ADDITION, "no_file.ttl", "text/turtle")) {
        const char* what;
        size_t whatLength;
        bool isRDFoxException;
        const char* exceptionName;
        size_t exceptionNameLength;
        const char* message;
        size_t messageLength;
        CRDFoxServer_getLastException(&what, &whatLength, &isRDFoxException, &exceptionName, &exceptionNameLength, &message, &messageLength);
        printf("Exception:\n");
        printf("Name: %s\n", exceptionName);
        printf("What: %s\n", what);
    }

    handleCRDFoxException(CDataStoreConnection_importDataFromFile(dataStoreConnection, NULL, CUpdateType_ADDITION, "lubm1.ttl", "text/turtle"));

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
    printf("Number of tuples after import: %zu\n", getTriplesCount(dataStoreConnection, "all"));

    // SPARQL queries can be evaluated in several ways. One option is to have the query result be written to
    // an output stream in one of the supported formats.
    struct COutputStream outputStream = { NULL, &stdoutOutputStreamFlush, &stdoutOutputStreamWrite };
    CStatementResult statementResult;
    handleCRDFoxException(CDataStoreConnection_evaluateStatement(dataStoreConnection, "SELECT DISTINCT ?Y WHERE { ?X ?Y ?Z }", 37, CParameters_getEmptyParameters(), &outputStream, "application/sparql-results+json", &statementResult));
    printf("Query produced %zu answers.\n", statementResult.numberOfQueryAnswers);

    // We now add the ontology and the custom rules to the data.

    // In this example, the rules are kept in a file separate from the ontology. JRDFox supports
    // SWRL rules, so it is possible to store the rules into the OWL ontology.

    printf("Adding the ontology to the store...\n");
    handleCRDFoxException(CDataStoreConnection_importDataFromFile(dataStoreConnection, NULL, CUpdateType_ADDITION, "univ-bench.owl", "text/owl-functional"));

    printf("Importing rules from a file...\n");
    handleCRDFoxException(CDataStoreConnection_importDataFromFile(dataStoreConnection, NULL, CUpdateType_ADDITION, "additional-rules.txt", ""));

    printf("Number of tuples after materialization: %zu\n", getTriplesCount(dataStoreConnection, "all"));

    // We now evaluate the same query as before, but we do so using a cursor, which provides us with
    // programmatic access to individual query results.

    CCursor* cursor = NULL;
    CDataStoreConnection_createCursor(dataStoreConnection, "SELECT DISTINCT ?Y WHERE { ?X ?Y ?Z }", 37, CParameters_getEmptyParameters(), &cursor);

    int numberOfRows = 0;
    printf("\n=======================================================================================\n");

    size_t arity;
    CCursor_getArity(cursor, &arity);
    const char* variableName;
    for (size_t termIndex = 0; termIndex < arity; ++termIndex) {
        if (termIndex != 0)
            printf("  ");
        // For each variable, we print the name.
        CCursor_getAnswerVariableName(cursor, termIndex, &variableName);
        printf("?%s", variableName);
    }
    printf("\n---------------------------------------------------------------------------------------\n");

    size_t multiplicity;
    // We iterate through the result tuples.
    for (CCursor_open(cursor, 0, &multiplicity); multiplicity != 0; CCursor_advance(cursor, &multiplicity)) {
        ++numberOfRows;
        // We iterate through the terms of each tuple.
        for (size_t termIndex = 0; termIndex < arity; ++termIndex) {
            if (termIndex != 0)
                printf("  ");
            // For each term, we retrieve the lexical form and the data type of the term.
            CDatatypeID datatypeID;
            char lexicalFormBuffer[1024];
            size_t lexicalFormSize = 0;
            bool resourceResolved = false;
            CCursor_appendResourceLexicalForm(cursor, termIndex, lexicalFormBuffer, sizeof(lexicalFormBuffer), &lexicalFormSize, &datatypeID, &resourceResolved);
            if (lexicalFormSize >= sizeof(lexicalFormBuffer)) {
                CCursor_getAnswerVariableName(cursor, termIndex, &variableName);
                printf("Warning: the lexical form for the term bound to variable ?%s on row %d was truncated from %zu to %zu characters.\n", variableName, numberOfRows, lexicalFormSize, sizeof(lexicalFormBuffer) - 1);
                lexicalFormBuffer[sizeof(lexicalFormBuffer) - 1] = '\0';
            }
            else
                lexicalFormBuffer[lexicalFormSize] = '\0';
            printf("%s", lexicalFormBuffer);
        }
        printf("\n");
    }
    printf("---------------------------------------------------------------------------------------\n");
    printf("  The number of rows returned: %d\n", numberOfRows);
    printf("=======================================================================================\n\n");

    // RDFox supports incremental reasoning. One can import facts into the store incrementally by
    // calling importDataFromFile(dataStoreConnection, ...) with additional argument CUpdateType_ADDITION.
    printf("Importing triples for incremental reasoning...\n");
    CDataStoreConnection_importDataFromFile(dataStoreConnection, NULL, CUpdateType_ADDITION, "lubm1-new.ttl", "text/turtle");

    // Adding the rules/facts changes the number of triples. Note that the store is updated incrementally.
    printf("Number of tuples after addition: %zu\n\n", getTriplesCount(dataStoreConnection, "all"));

    // Content can be imported from buffers or from streams as well as from file. Here we will add one triple stored
    // in a c string using CDataStoreConnection_importDataFromBuffer and then remove the same content using an in-memory
    // implementation of CInputStream for the same c string. Each time we will count the triples for comparison.
    printf("Adding hard-coded triple...\n");
    const char* hardCodedTriple = "<http://example.com/subject> <http://example.com/predicate> <http://example.com/object> .";
    const size_t hardCodedTripleLength = strlen(hardCodedTriple);
    CDataStoreConnection_importDataFromBuffer(dataStoreConnection, NULL, CUpdateType_ADDITION, hardCodedTriple, hardCodedTripleLength, "text/turtle");
    printf("Number of tuples after adding hard-coded triple: %zu\n\n", getTriplesCount(dataStoreConnection, "all"));
    printf("Removing hard-coded triple...\n");
    struct MemoryInputStreamContext memoryInputStreamContext = { hardCodedTriple, hardCodedTriple + hardCodedTripleLength, hardCodedTriple };
    struct CInputStream inputStream = { &memoryInputStreamContext, &memoryInputStreamRewind, &memoryInputStreamRead };
    CDataStoreConnection_importData(dataStoreConnection, NULL, CUpdateType_DELETION, &inputStream, "https://rdfox.com/default-base-iri/", "text/turtle");
    printf("Number of tuples after removing hard-coded triple: %zu\n\n", getTriplesCount(dataStoreConnection, "all"));

    // One can export the facts from the current store into a file as follows.
    printf("Exporting facts to file 'final-facts.ttl'...\n");
    fflush(stdout);
    CParameters* parameters = NULL;
    CParameters_newEmptyParameters(&parameters);
    CParameters_setString(parameters, "fact-domain", "all");
    CDataStoreConnection_exportDataToFile(dataStoreConnection, "final-facts.ttl", "application/n-triples", parameters);
    CParameters_destroy(parameters);

    CDataStoreConnection_destroy(dataStoreConnection);

    printf("This is the end of the example!\n");
    return 0;
}
