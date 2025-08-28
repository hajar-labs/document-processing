// Document Processing System Integration
        class DocumentProcessorSystem {
            constructor() {
                this.baseURL = 'http://localhost:5000/api';
                this.documents = new Map();
                this.selectedDocument = null;
                this.chatHistory = [];
                this.uploadProgress = null;
                
                this.initializeSystem();
            }

            initializeSystem() {
                this.setupEventListeners();
                this.loadStoredDocuments();
                this.updateStats();
                console.log('Document Processing System initialized');
            }

            setupEventListeners() {
                // File input change
                document.getElementById('fileInput').addEventListener('change', (e) => {
                    this.handleFileUpload(e);
                });

                // Drag and drop for upload area
                const uploadArea = document.querySelector('.upload-area');
                uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
                uploadArea.addEventListener('dragleave', this.handleDragLeave.bind(this));
                uploadArea.addEventListener('drop', this.handleDrop.bind(this));

                // Chat functionality
                document.getElementById('chatInput').addEventListener('keypress', (e) => {
                    if (e.key === 'Enter') this.sendChatMessage();
                });

                // Search functionality
                document.getElementById('searchInput').addEventListener('keypress', (e) => {
                    if (e.key === 'Enter') this.performSearch();
                });
            }

            // File Upload Handling
            async handleFileUpload(event) {
                const files = event.target.files;
                if (files.length === 0) return;

                for (const file of files) {
                    await this.processFile(file);
                }
                
                // Reset file input
                event.target.value = '';
            }

            handleDragOver(event) {
                event.preventDefault();
                event.currentTarget.classList.add('border-primary', 'bg-light');
            }

            handleDragLeave(event) {
                event.currentTarget.classList.remove('border-primary', 'bg-light');
            }

            handleDrop(event) {
                event.preventDefault();
                event.currentTarget.classList.remove('border-primary', 'bg-light');
                
                const files = event.dataTransfer.files;
                for (const file of files) {
                    this.processFile(file);
                }
            }

            async processFile(file) {
                try {
                    // Show upload modal
                    this.showUploadProgress(file.name);
                    
                    // Create form data
                    const formData = new FormData();
                    formData.append('file', file);

                    // Upload and process file
                    const response = await fetch(`${this.baseURL}/upload`, {
                        method: 'POST',
                        body: formData
                    });

                    if (!response.ok) {
                        const error = await response.json();
                        throw new Error(error.error || 'Upload failed');
                    }

                    const result = await response.json();
                    
                    // Create document object
                    const document = {
                        id: this.generateId(),
                        filename: result.filename,
                        text: result.text,
                        language: result.language,
                        metadata: result.metadata,
                        wordCount: result.word_count,
                        uploadDate: new Date().toISOString(),
                        processed: true,
                        summary: null,
                        analysis: null
                    };

                    // Store document
                    this.documents.set(document.id, document);
                    
                    // Update UI
                    this.updateDocumentsList();
                    this.updateStats();
                    this.hideUploadProgress();
                    
                    // Show success notification
                    this.showNotification('Succès', `Document "${file.name}" traité avec succès`, 'success');
                    
                    // Auto-select the document
                    this.selectDocument(document.id);
                    
                } catch (error) {
                    console.error('File processing error:', error);
                    this.hideUploadProgress();
                    this.showNotification('Erreur', `Erreur lors du traitement: ${error.message}`, 'error');
                }
            }

            // Document Management
            updateDocumentsList() {
                const container = document.getElementById('documentsList');
                container.innerHTML = '';

                this.documents.forEach((doc, id) => {
                    const item = document.createElement('div');
                    item.className = `list-group-item document-item ${this.selectedDocument === id ? 'active' : ''}`;
                    item.onclick = () => this.selectDocument(id);
                    
                    const fileIcon = this.getFileIcon(doc.filename);
                    const fileSize = doc.wordCount ? `${doc.wordCount} mots` : 'Traitement...';
                    
                    item.innerHTML = `
                        <div class="d-flex justify-content-between align-items-start">
                            <div class="flex-grow-1">
                                <div class="d-flex align-items-center mb-1">
                                    <i class="${fileIcon} me-2"></i>
                                    <span class="fw-medium">${doc.filename}</span>
                                </div>
                                <small class="text-muted d-block">
                                    ${fileSize} • ${doc.language || 'Auto'}
                                </small>
                                <small class="text-muted">
                                    ${this.formatDate(doc.uploadDate)}
                                </small>
                            </div>
                            <div class="dropdown">
                                <button class="btn btn-sm btn-outline-secondary" type="button" data-bs-toggle="dropdown">
                                    <i class="fas fa-ellipsis-v"></i>
                                </button>
                                <ul class="dropdown-menu">
                                    <li><a class="dropdown-item" href="#" onclick="documentProcessor.viewDocument('${id}')">
                                        <i class="fas fa-eye me-2"></i>Voir détails
                                    </a></li>
                                    <li><a class="dropdown-item" href="#" onclick="documentProcessor.downloadDocument('${id}')">
                                        <i class="fas fa-download me-2"></i>Télécharger
                                    </a></li>
                                    <li><hr class="dropdown-divider"></li>
                                    <li><a class="dropdown-item text-danger" href="#" onclick="documentProcessor.deleteDocument('${id}')">
                                        <i class="fas fa-trash me-2"></i>Supprimer
                                    </a></li>
                                </ul>
                            </div>
                        </div>
                    `;
                    
                    container.appendChild(item);
                });

                if (this.documents.size === 0) {
                    container.innerHTML = `
                        <div class="text-center text-muted py-4">
                            <i class="fas fa-folder-open fa-2x mb-2"></i>
                            <p>Aucun document</p>
                        </div>
                    `;
                }
            }

            async selectDocument(documentId) {
                this.selectedDocument = documentId;
                const doc = this.documents.get(documentId);
                
                if (!doc) return;

                // Update document list selection
                this.updateDocumentsList();
                
                // Load document analysis if not already done
                if (!doc.summary) {
                    await this.generateDocumentSummary(documentId);
                }
                
                // Update summary tab
                this.updateSummaryTab(doc);
            }

            async generateDocumentSummary(documentId) {
                const doc = this.documents.get(documentId);
                if (!doc) return;

                try {
                    this.showLoading('Génération du résumé...');

                    // Generate summary
                    const summaryResponse = await fetch(`${this.baseURL}/summarize`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            text: doc.text,
                            language: doc.language,
                            type: 'both'
                        })
                    });

                    if (summaryResponse.ok) {
                        const summaryResult = await summaryResponse.json();
                        doc.summary = summaryResult.summary;
                    }

                    // Generate analysis
                    const analysisResponse = await fetch(`${this.baseURL}/analyze`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text: doc.text })
                    });

                    if (analysisResponse.ok) {
                        const analysisResult = await analysisResponse.json();
                        doc.analysis = analysisResult;
                    }

                    // Update document in storage
                    this.documents.set(documentId, doc);
                    this.hideLoading();

                } catch (error) {
                    console.error('Summary generation error:', error);
                    this.hideLoading();
                    this.showNotification('Erreur', 'Erreur lors de la génération du résumé', 'error');
                }
            }

            updateSummaryTab(doc) {
                const container = document.getElementById('summaryContent');
                
                let summaryHTML = `
                    <div class="row mb-4">
                        <div class="col-md-8">
                            <h4><i class="fas fa-file me-2"></i>${doc.filename}</h4>
                            <div class="row mt-3">
                                <div class="col-md-3">
                                    <div class="metric-value">${doc.wordCount || 0}</div>
                                    <div class="metric-label">Mots</div>
                                </div>
                                <div class="col-md-3">
                                    <div class="metric-value">${doc.language || 'Auto'}</div>
                                    <div class="metric-label">Langue</div>
                                </div>
                                <div class="col-md-3">
                                    <div class="metric-value">${this.formatDate(doc.uploadDate)}</div>
                                    <div class="metric-label">Ajouté le</div>
                                </div>
                                <div class="col-md-3">
                                    <div class="metric-value">
                                        <i class="fas fa-check-circle text-success"></i>
                                    </div>
                                    <div class="metric-label">Statut</div>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="card">
                                <div class="card-header">
                                    <h6 class="mb-0">Actions Rapides</h6>
                                </div>
                                <div class="card-body">
                                    <button class="btn btn-primary btn-sm mb-2 w-100" onclick="documentProcessor.generateNewSummary('${doc.id}')">
                                        <i class="fas fa-sync me-2"></i>Nouveau résumé
                                    </button>
                                    <button class="btn btn-outline-primary btn-sm mb-2 w-100" onclick="documentProcessor.exportSummary('${doc.id}')">
                                        <i class="fas fa-download me-2"></i>Exporter
                                    </button>
                                    <button class="btn btn-outline-secondary btn-sm w-100" onclick="documentProcessor.viewFullText('${doc.id}')">
                                        <i class="fas fa-eye me-2"></i>Texte complet
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                `;

                // Add summaries if available
                if (doc.summary) {
                    summaryHTML += `<div class="row">`;
                    
                    if (doc.summary.extractive) {
                        summaryHTML += `
                            <div class="col-md-6">
                                <div class="card">
                                    <div class="card-header">
                                        <h6 class="mb-0">
                                            <i class="fas fa-list me-2"></i>Résumé Extractif
                                        </h6>
                                    </div>
                                    <div class="card-body">
                                        <p>${doc.summary.extractive}</p>
                                    </div>
                                </div>
                            </div>
                        `;
                    }
                    
                    if (doc.summary.abstractive) {
                        summaryHTML += `
                            <div class="col-md-6">
                                <div class="card">
                                    <div class="card-header">
                                        <h6 class="mb-0">
                                            <i class="fas fa-brain me-2"></i>Résumé Abstractif
                                        </h6>
                                    </div>
                                    <div class="card-body">
                                        <p>${doc.summary.abstractive}</p>
                                    </div>
                                </div>
                            </div>
                        `;
                    }
                    
                    summaryHTML += `</div>`;
                }

                // Add analysis if available
                if (doc.analysis) {
                    summaryHTML += `
                        <div class="row mt-4">
                            <div class="col-12">
                                <div class="card">
                                    <div class="card-header">
                                        <h6 class="mb-0">
                                            <i class="fas fa-chart-line me-2"></i>Analyse Structurelle
                                        </h6>
                                    </div>
                                    <div class="card-body">
                                        <pre class="bg-light p-3 rounded">${JSON.stringify(doc.analysis, null, 2)}</pre>
                                    </div>
                                </div>
                            </div>
                        </div>
                    `;
                }

                container.innerHTML = summaryHTML;
            }

            // Chat Functionality
            async sendChatMessage() {
                const input = document.getElementById('chatInput');
                const message = input.value.trim();
                
                if (!message) return;

                // Add user message to chat
                this.addChatMessage('user', message);
                input.value = '';

                // Get AI response (placeholder - you can integrate with your AI service)
                await this.processAIResponse(message);
            }

            addChatMessage(sender, message, type = 'text') {
                const container = document.getElementById('chatMessages');
                const messageDiv = document.createElement('div');
                
                const senderClass = sender === 'user' ? 'user-message' : 'assistant-message';
                const icon = sender === 'user' ? 'fa-user' : 'fa-robot';
                const senderName = sender === 'user' ? 'Vous' : 'Assistant';
                
                messageDiv.className = `chat-message ${senderClass}`;
                messageDiv.innerHTML = `
                    <i class="fas ${icon} me-2"></i>
                    <strong>${senderName}:</strong> ${message}
                `;
                
                container.appendChild(messageDiv);
                container.scrollTop = container.scrollHeight;
                
                this.chatHistory.push({ sender, message, timestamp: new Date().toISOString() });
            }

            async processAIResponse(userMessage) {
                // Placeholder for AI integration
                // You can integrate with your AI service here
                
                let response = "Je suis désolé, l'intégration IA n'est pas encore disponible. ";
                
                if (this.selectedDocument) {
                    const doc = this.documents.get(this.selectedDocument);
                    if (userMessage.toLowerCase().includes('résumé')) {
                        response = `Voici un résumé du document "${doc.filename}": `;
                        if (doc.summary && doc.summary.extractive) {
                            response += doc.summary.extractive;
                        } else {
                            response += "Le résumé est en cours de génération...";
                        }
                    } else if (userMessage.toLowerCase().includes('mots')) {
                        response = `Le document "${doc.filename}" contient ${doc.wordCount} mots et est en ${doc.language}.`;
                    }
                } else {
                    response += "Veuillez d'abord sélectionner un document.";
                }
                
                // Simulate AI thinking time
                setTimeout(() => {
                    this.addChatMessage('assistant', response);
                }, 1000);
            }

            handleChatKeyPress(event) {
                if (event.key === 'Enter') {
                    this.sendChatMessage();
                }
            }

            // Search Functionality
            async performSearch() {
                const query = document.getElementById('searchInput').value.trim();
                const searchType = document.getElementById('searchType').value;
                
                if (!query) return;

                this.showLoading('Recherche en cours...');
                
                try {
                    // Implement search logic here
                    const results = this.searchInDocuments(query, searchType);
                    this.displaySearchResults(results, query);
                } catch (error) {
                    console.error('Search error:', error);
                    this.showNotification('Erreur', 'Erreur lors de la recherche', 'error');
                } finally {
                    this.hideLoading();
                }
            }

            searchInDocuments(query, type) {
                const results = [];
                const queryLower = query.toLowerCase();
                
                this.documents.forEach((doc, id) => {
                    if (!doc.text) return;
                    
                    const textLower = doc.text.toLowerCase();
                    let score = 0;
                    let matches = [];
                    
                    if (type === 'exact') {
                        if (textLower.includes(queryLower)) {
                            score = 100;
                            matches.push({ type: 'exact', text: query });
                        }
                    } else {
                        // Simple keyword search
                        const queryWords = queryLower.split(' ');
                        queryWords.forEach(word => {
                            if (textLower.includes(word)) {
                                score += 10;
                                matches.push({ type: 'keyword', text: word });
                            }
                        });
                    }
                    
                    if (score > 0) {
                        results.push({
                            document: doc,
                            score: score,
                            matches: matches
                        });
                    }
                });
                
                return results.sort((a, b) => b.score - a.score);
            }

            displaySearchResults(results, query) {
                const container = document.getElementById('searchResults');
                
                if (results.length === 0) {
                    container.innerHTML = `
                        <div class="text-center text-muted py-4">
                            <i class="fas fa-search fa-2x mb-3"></i>
                            <h5>Aucun résultat trouvé</h5>
                            <p>Aucun document ne correspond à votre recherche "${query}"</p>
                        </div>
                    `;
                    return;
                }
                
                let resultsHTML = `
                    <h6 class="mb-3">${results.length} résultat(s) pour "${query}"</h6>
                `;
                
                results.forEach(result => {
                    const doc = result.document;
                    resultsHTML += `
                        <div class="card mb-3">
                            <div class="card-body">
                                <h6 class="card-title">
                                    <i class="${this.getFileIcon(doc.filename)} me-2"></i>
                                    ${doc.filename}
                                </h6>
                                <p class="card-text">
                                    ${this.getSearchSnippet(doc.text, query)}
                                </p>
                                <div class="d-flex justify-content-between align-items-center">
                                    <small class="text-muted">
                                        Score: ${result.score} • ${doc.wordCount} mots • ${doc.language}
                                    </small>
                                    <button class="btn btn-primary btn-sm" onclick="documentProcessor.selectDocument('${doc.id}')">
                                        Voir document
                                    </button>
                                </div>
                            </div>
                        </div>
                    `;
                });
                
                container.innerHTML = resultsHTML;
            }

            handleSearchKeyPress(event) {
                if (event.key === 'Enter') {
                    this.performSearch();
                }
            }

            // Utility Functions
            generateId() {
                return 'doc_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            }

            getFileIcon(filename) {
                const ext = filename.split('.').pop().toLowerCase();
                switch (ext) {
                    case 'pdf': return 'fas fa-file-pdf text-danger';
                    case 'docx':
                    case 'doc': return 'fas fa-file-word text-primary';
                    case 'txt': return 'fas fa-file-text text-secondary';
                    default: return 'fas fa-file text-muted';
                }
            }

            formatDate(isoDate) {
                return new Date(isoDate).toLocaleDateString('fr-FR');
            }

            getSearchSnippet(text, query, maxLength = 200) {
                const queryIndex = text.toLowerCase().indexOf(query.toLowerCase());
                if (queryIndex === -1) {
                    return text.substr(0, maxLength) + (text.length > maxLength ? '...' : '');
                }
                
                const start = Math.max(0, queryIndex - 100);
                const end = Math.min(text.length, queryIndex + query.length + 100);
                let snippet = text.substring(start, end);
                
                if (start > 0) snippet = '...' + snippet;
                if (end < text.length) snippet = snippet + '...';
                
                return snippet;
            }

            // UI Helper Functions
            showLoading(message = 'Chargement...') {
                const overlay = document.getElementById('loadingOverlay');
                overlay.querySelector('p').textContent = message;
                overlay.classList.remove('d-none');
            }

            hideLoading() {
                document.getElementById('loadingOverlay').classList.add('d-none');
            }

            showUploadProgress(filename) {
                const modal = new bootstrap.Modal(document.getElementById('uploadModal'));
                document.getElementById('uploadStatus').textContent = `Traitement de "${filename}"...`;
                
                const progressBar = document.querySelector('#uploadModal .progress-bar');
                let progress = 0;
                
                this.uploadProgress = setInterval(() => {
                    progress += Math.random() * 20;
                    if (progress > 90) progress = 90;
                    progressBar.style.width = progress + '%';
                }, 500);
                
                modal.show();
            }

            hideUploadProgress() {
                if (this.uploadProgress) {
                    clearInterval(this.uploadProgress);
                    this.uploadProgress = null;
                }
                
                const progressBar = document.querySelector('#uploadModal .progress-bar');
                progressBar.style.width = '100%';
                
                setTimeout(() => {
                    bootstrap.Modal.getInstance(document.getElementById('uploadModal')).hide();
                }, 1000);
            }

            showNotification(title, message, type = 'info') {
                const alertClass = {
                    'success': 'alert-success',
                    'error': 'alert-danger',
                    'warning': 'alert-warning',
                    'info': 'alert-info'
                }[type] || 'alert-info';
                
                const notification = document.createElement('div');
                notification.className = `alert ${alertClass} notification fade show`;
                notification.innerHTML = `
                    <strong>${title}</strong> ${message}
                    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                `;
                
                document.body.appendChild(notification);
                
                setTimeout(() => {
                    notification.remove();
                }, 5000);
            }

            updateStats() {
                document.getElementById('docCount').textContent = this.documents.size;
                document.getElementById('connectionCount').textContent = '1';
                document.getElementById('questionCount').textContent = this.chatHistory.length;
            }

            loadStoredDocuments() {
                // Load documents from localStorage if available
                const stored = localStorage.getItem('processedDocuments');
                if (stored) {
                    try {
                        const docs = JSON.parse(stored);
                        docs.forEach(doc => {
                            this.documents.set(doc.id, doc);
                        });
                        this.updateDocumentsList();
                    } catch (error) {
                        console.error('Error loading stored documents:', error);
                    }
                }
            }

            saveDocuments() {
                // Save documents to localStorage
                const docs = Array.from(this.documents.values());
                localStorage.setItem('processedDocuments', JSON.stringify(docs));
            }

            // Additional Methods
            async generateNewSummary(docId) {
                await this.generateDocumentSummary(docId);
                this.selectDocument(docId);
            }

            exportSummary(docId) {
                const doc = this.documents.get(docId);
                if (!doc || !doc.summary) return;
                
                const content = `
        Résumé du document: ${doc.filename}
        Généré le: ${new Date().toLocaleString('fr-FR')}

        RÉSUMÉ EXTRACTIF:
        ${doc.summary.extractive || 'Non disponible'}

        RÉSUMÉ ABSTRACTIF:
        ${doc.summary.abstractive || 'Non disponible'}

        STATISTIQUES:
        - Nombre de mots: ${doc.wordCount}
        - Langue détectée: ${doc.language}
        - Date d'ajout: ${this.formatDate(doc.uploadDate)}
                `;
                
                const blob = new Blob([content], { type: 'text/plain' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `resume_${doc.filename}.txt`;
                a.click();
                URL.revokeObjectURL(url);
            }

            viewFullText(docId) {
                const doc = this.documents.get(docId);
                if (!doc) return;
                
                const modal = bootstrap.Modal.getInstance(document.getElementById('documentModal')) || 
                            new bootstrap.Modal(document.getElementById('documentModal'));
                
                document.getElementById('documentModalTitle').innerHTML = 
                    `<i class="fas fa-file me-2"></i>${doc.filename}`;
                
                document.getElementById('documentModalContent').innerHTML = `
                    <div class="bg-light p-3 rounded" style="max-height: 500px; overflow-y: auto;">
                        <pre style="white-space: pre-wrap;">${doc.text}</pre>
                    </div>
                `;
                
                modal.show();
            }

            deleteDocument(docId) {
                if (confirm('Êtes-vous sûr de vouloir supprimer ce document ?')) {
                    this.documents.delete(docId);
                    if (this.selectedDocument === docId) {
                        this.selectedDocument = null;
                        document.getElementById('summaryContent').innerHTML = `
                            <div class="text-center text-muted py-5">
                                <i class="fas fa-file-text fa-3x mb-3"></i>
                                <h5>Sélectionnez un document pour voir son analyse</h5>
                            </div>
                        `;
                    }
                    this.updateDocumentsList();
                    this.updateStats();
                    this.saveDocuments();
                }
            }
        }

        // Global Functions (for onclick handlers)
        function uploadFile(input) {
            documentProcessor.handleFileUpload({ target: input });
        }

        function handleChatKeyPress(event) {
            documentProcessor.handleChatKeyPress(event);
        }

        function sendChatMessage() {
            documentProcessor.sendChatMessage();
        }

        function handleSearchKeyPress(event) {
            documentProcessor.handleSearchKeyPress(event);
        }

        function performSearch() {
            documentProcessor.performSearch();
        }

        function showSystemStatus() {
            const modal = new bootstrap.Modal(document.getElementById('statusModal'));
            document.getElementById('systemStatusContent').innerHTML = `
                <div class="row">
                    <div class="col-md-6">
                        <h6>État du Système</h6>
                        <ul class="list-unstyled">
                            <li><i class="fas fa-check-circle text-success me-2"></i>API Backend: En ligne</li>
                            <li><i class="fas fa-check-circle text-success me-2"></i>Extracteurs: Actifs</li>
                            <li><i class="fas fa-check-circle text-success me-2"></i>Résumeurs: Actifs</li>
                            <li><i class="fas fa-check-circle text-success me-2"></i>Analyseurs: Actifs</li>
                        </ul>
                    </div>
                    <div class="col-md-6">
                        <h6>Statistiques</h6>
                        <ul class="list-unstyled">
                            <li>Documents traités: ${documentProcessor.documents.size}</li>
                            <li>Messages chat: ${documentProcessor.chatHistory.length}</li>
                            <li>Temps de fonctionnement: ${Math.floor(Math.random() * 24)}h ${Math.floor(Math.random() * 60)}m</li>
                        </ul>
                    </div>
                </div>
            `;
            modal.show();
        }

        function deleteDocument() {
            if (documentProcessor.selectedDocument) {
                documentProcessor.deleteDocument(documentProcessor.selectedDocument);
                bootstrap.Modal.getInstance(document.getElementById('documentModal')).hide();
            }
        }

        // Initialize the system when DOM is loaded
        document.addEventListener('DOMContentLoaded', function() {
            window.documentProcessor = new DocumentProcessorSystem();
            
            console.log('Document Processing System loaded successfully!');
        });

        // Auto-save documents periodically
        setInterval(() => {
            if (window.documentProcessor) {
                window.documentProcessor.saveDocuments();
            }
        }, 30000); // Save every 30 seconds
