print("Initializing MongoDB...");

const adminDb = db.getSiblingDB("admin");

// Use process.env to access environment variables
const dbName = process.env["MONGO_INITDB_DATABASE"];
const appUser = process.env["MONGO_APP_USERNAME"];
const appPassword = process.env["MONGO_APP_PASSWORD"];

if (!dbName || !appUser || !appPassword) {
    print("Missing required environment variables. Exiting...");
    quit(1);
}

// Authenticate as root user
adminDb.auth(process.env["MONGO_INITDB_ROOT_USERNAME"], process.env["MONGO_INITDB_ROOT_PASSWORD"]);

// Check if the database already exists
if (adminDb.getMongo().getDBNames().indexOf(dbName) === -1) {
    print(`Database ${dbName} not found. Creating database and user...`);
    
    const dbInstance = db.getSiblingDB(dbName);
    dbInstance.createUser({
        user: appUser,
        pwd: appPassword,
        roles: [{ role: "readWrite", db: dbName }]
    });
    
    print(`User ${appUser} created successfully.`);
    
    print("Creating required collections...");
    dbInstance.createCollection("articles", {
        validator: {
            $jsonSchema: {
                bsonType: "object",
                required: ["title", "link", "image", "pubDate", "source"],
                properties: {
                    title: {
                        bsonType: "string",
                        description: "Title of the article"
                    },
                    link: {
                        bsonType: "string",
                        description: "URL to the article"
                    },
                    image: {
                        bsonType: "binData",
                        description: "The image of the article"
                    },
                    image_url: {
                        bsonType: "string",
                        description: "The URL for the image"
                    },
                    pubDate: {
                        bsonType: "date",
                        description: "Publication date of the article"
                    },
                    source: {
                        bsonType: "string",
                        description: "Source of the article"
                    },
                    summary: {
                        bsonType: "string",
                        description: "Summary or description of the article"
                    }
                }
            }
        }
    });
    print("Collections created successfully.");
} else {
    print(`Database ${dbName} already exists. Validating collections...`);
    
    const dbInstance = db.getSiblingDB(dbName);
    const requiredCollection = "articles";
    const existingCollections = dbInstance.getCollectionNames();
    
        if (!existingCollections.includes(requiredCollection)) {
            print(`Collection ${collection} is missing. Creating it...`);
            dbInstance.createCollection("articles", {
                validator: {
                    $jsonSchema: {
                        bsonType: "object",
                        required: ["title", "link", "image", "pubDate", "source"],
                        properties: {
                            title: {
                                bsonType: "string",
                                description: "Title of the article"
                            },
                            link: {
                                bsonType: "string",
                                description: "URL to the article"
                            },
                            image: {
                                bsonType: "binData",
                                description: "The image of the article"
                            },
                            pubDate: {
                                bsonType: "date",
                                description: "Publication date of the article"
                            },
                            source: {
                                bsonType: "string",
                                description: "Source of the article"
                            },
                            summary: {
                                bsonType: "string",
                                description: "Summary or description of the article"
                            }
                        }
                    }
                }
            });
        }
    print("Database validation completed.");
    }

print("MongoDB initialization complete.");