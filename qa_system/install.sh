git clone git://github.com/lendormi/stanford-corenlp-python.git
cd stanford-corenlp-python
/usr/local/stow/python/amd64_linux26/python-2.7.3/bin/pip install -t . pexpect unidecode
wget http://nlp.stanford.edu/software/stanford-corenlp-full-2015-04-20.zip
unzip stanford-corenlp-full-2015-04-20.zip
cd ..
cp qa_system.zip stanford-corenlp-python/
cd stanford-corenlp-python
unzip qa_system.zip
rm -r qa_system.zip
cd ..


