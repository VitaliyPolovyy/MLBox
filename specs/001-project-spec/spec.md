# Feature Specification: MLBox Multi-Service ML Platform

**Feature Branch**: `001-project-spec`  
**Created**: 2025-11-25  
**Status**: Draft  
**Input**: User description: "analyze project and write spec"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Deploy and Manage Multiple ML Services (Priority: P1)

Product managers need a platform solution that can host and manage multiple independent ML services, each serving different business needs.

**Why this priority**: This is the core platform capability - MLBox must support multiple services (currently LabelGuard and Peanuts) with the ability to add more services in the future.

**Independent Test**: Can be fully tested by verifying that both LabelGuard and Peanuts services run independently on the same platform, each with their own endpoints, processing logic, and results, without interfering with each other.

**Acceptance Scenarios**:

1. **Given** MLBox platform is deployed, **When** multiple services are configured, **Then** each service operates independently with its own endpoints and processing logic
2. **Given** multiple services are running, **When** requests are sent to different services, **Then** each service processes requests correctly without cross-service interference
3. **Given** a new service needs to be added, **When** it follows platform conventions, **Then** it can be deployed alongside existing services without breaking them

---

### User Story 2 - Process Product Label for Verification via LabelGuard Service (Priority: P1)

Users need to verify product labels meet regulatory and quality requirements by uploading label images and receiving validation results through the LabelGuard service.

**Why this priority**: This is the core functionality of the LabelGuard service, providing automated label verification that replaces manual inspection.

**Independent Test**: Can be fully tested by uploading a product label image via HTTP POST to `/labelguard/upload`, then calling `/labelguard/analyze` endpoint, and verifying that the system returns text blocks, categories, languages, and validation results.

**Acceptance Scenarios**:

1. **Given** a user has a product label image, **When** they upload it to the LabelGuard service, **Then** the system stores the image and returns a file path for analysis
2. **Given** an uploaded label image exists, **When** the user requests analysis, **Then** the system extracts text blocks, categorizes them, detects languages, and validates against rules
3. **Given** a label image with corrections needed, **When** the user provides corrected text blocks, **Then** the system re-analyzes with the corrections and returns updated validation results
4. **Given** a label image is processed, **When** the user views results, **Then** they can see detected text blocks, categories, languages, and validation status with visual markers

---

### User Story 3 - Process Peanut Quality Analysis via Peanuts Service (Priority: P1)

Users need to analyze peanut quality by processing images and receiving classification results with detailed measurements through the Peanuts service.

**Why this priority**: This is the core functionality of the Peanuts service, providing automated quality control for peanut products.

**Independent Test**: Can be fully tested by sending a peanut image via HTTP POST to `/peanuts/process_image` with JSON metadata, and verifying that the system returns detection results, classifications, and generates an Excel report.

**Acceptance Scenarios**:

1. **Given** a user has a peanut image, **When** they submit it with processing metadata, **Then** the system detects individual peanuts, classifies them, and measures their characteristics
2. **Given** peanut processing is complete, **When** the system generates results, **Then** it creates an Excel file with measurements and classifications
3. **Given** processing results are ready, **When** the system completes, **Then** it automatically sends results back to the ERP system via HTTP POST
4. **Given** multiple peanut images are submitted, **When** they are processed in batch, **Then** each image is processed independently and results are returned for all

---

### User Story 4 - Monitor Platform and Service Health (Priority: P2)

Operators need to monitor service health, view logs, and access debugging artifacts to ensure services are running correctly.

**Why this priority**: Essential for operations and troubleshooting, but secondary to core processing functionality.

**Independent Test**: Can be fully tested by calling health check endpoints and verifying that services respond with status information, and by accessing artifact storage to view debug outputs.

**Acceptance Scenarios**:

1. **Given** a service is running, **When** an operator checks the health endpoint, **Then** the system returns service status and availability
2. **Given** debug mode is enabled, **When** services process requests, **Then** intermediate processing artifacts are saved for troubleshooting
3. **Given** artifacts are generated, **When** operators access the artifact viewer, **Then** they can view processing steps and debug information

---

### Edge Cases

**Platform Edge Cases**:

- What happens when a new service is added while other services are running?
- How does the platform handle requests sent to non-existent services?
- What happens when one service fails - does it affect other services?
- How does the platform handle resource contention when multiple services process requests simultaneously?
- What happens when the platform runs out of storage for artifacts?

**LabelGuard Service Edge Cases**:

- What happens when an uploaded image is corrupted or in an unsupported format?
- How does the system handle label images with no detectable text?
- What happens when OCR fails to extract text from a label?
- How does the system handle multilingual labels with mixed languages?
- How does the system handle concurrent requests to the LabelGuard service?

**Peanuts Service Edge Cases**:

- What happens when peanut detection finds no peanuts in an image?
- How does the system handle batch processing when some images fail?
- What happens when the ERP endpoint is unreachable for result delivery?
- How does the system handle concurrent requests to the Peanuts service?
- What happens when model files are missing or corrupted?
- How does the system handle requests that exceed maximum image size?

## Requirements *(mandatory)*

### Functional Requirements

**Platform Requirements**:

- **FR-001**: Platform MUST host multiple independent ML services simultaneously
- **FR-002**: Platform MUST provide unified deployment infrastructure for all services
- **FR-003**: Platform MUST isolate services so they operate independently without interference
- **FR-004**: Platform MUST provide health check endpoints for service monitoring
- **FR-005**: Platform MUST save debug artifacts when debug mode is enabled, organized by service
- **FR-006**: Platform MUST validate all incoming requests using JSON schemas
- **FR-007**: Platform MUST handle errors gracefully and return structured error responses
- **FR-008**: Platform MUST support CORS for web-based clients
- **FR-009**: Platform MUST log all operations with structured logging
- **FR-010**: Platform MUST store artifacts in organized directories by service name

**LabelGuard Service Requirements**:

- **FR-011**: LabelGuard service MUST accept product label images via HTTP POST and store them for processing
- **FR-012**: LabelGuard service MUST extract text from label images using OCR technology
- **FR-013**: LabelGuard service MUST categorize extracted text blocks (ingredients, allergens, nutritional info, etc.)
- **FR-014**: LabelGuard service MUST detect languages in text blocks (supporting multiple languages)
- **FR-015**: LabelGuard service MUST validate labels against business rules (allergen detection, number validation, etc.)
- **FR-016**: LabelGuard service MUST allow users to correct detected text blocks and re-analyze

**Peanuts Service Requirements**:

- **FR-017**: Peanuts service MUST detect and classify individual peanuts in images
- **FR-018**: Peanuts service MUST measure peanut characteristics (size, shape, position)
- **FR-019**: Peanuts service MUST generate Excel reports with peanut analysis results
- **FR-020**: Peanuts service MUST automatically deliver results to ERP system via HTTP POST
- **FR-021**: Peanuts service MUST process multiple images in batch mode

### Key Entities *(include if feature involves data)*

**Platform Entities**:

- **ML Service**: Represents an independent ML service within the platform (e.g., LabelGuard, Peanuts), contains service-specific endpoints, processing logic, and configuration
- **Processing Request**: Represents a request to process data through a service, contains input data, metadata, and processing parameters
- **Artifact**: Represents a debug or intermediate file saved during processing, organized by service name and processing step

**LabelGuard Service Entities**:

- **Label Image**: Represents a product label image file uploaded for processing, contains image data and metadata
- **Text Block**: Represents a detected region of text on a label, contains text content, bounding box, category, and language
- **Label Processing Result**: Represents the complete analysis result for a label, contains text blocks, validation results, and visual markers

**Peanuts Service Entities**:

- **Peanut Image**: Represents a peanut analysis image file, contains image data and processing metadata
- **Peanut Detection**: Represents a detected peanut in an image, contains bounding box, mask, classification, and measurements
- **Peanut Processing Result**: Represents the complete analysis result for a peanut image, contains all detected peanuts and their classifications

## Success Criteria *(mandatory)*

### Measurable Outcomes

**Platform Outcomes**:

- **SC-001**: Platform successfully hosts at least 2 services simultaneously without interference
- **SC-002**: Platform maintains service availability with health check response time under 1 second
- **SC-003**: Platform handles 100 concurrent requests across all services without errors
- **SC-004**: Platform can add new services without breaking existing services

**LabelGuard Service Outcomes**:

- **SC-005**: LabelGuard service processes label images and returns analysis results within 30 seconds for standard label sizes
- **SC-006**: LabelGuard service correctly extracts text from 95% of readable label images
- **SC-007**: LabelGuard service correctly categorizes text blocks with 90% accuracy
- **SC-008**: LabelGuard service correctly identifies languages in multilingual labels for supported languages
- **SC-009**: LabelGuard service validates labels against business rules with 100% rule coverage

**Peanuts Service Outcomes**:

- **SC-010**: Peanuts service detects and classifies peanuts with 90% accuracy
- **SC-011**: Peanuts service processes batch requests of up to 8 images concurrently without degradation
- **SC-012**: Peanuts service successfully delivers results to ERP system in 99% of cases
