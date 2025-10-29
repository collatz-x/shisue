from typing import Optional


class MRIScanException(Exception):
    '''Base exception for all all MRI Scan Segmentation errors.'''
    def __init__(self, message: str, details: Optional[str] = None):
        '''
        Initializes the base exception.

        Args:
            message: The error message
            details: Optional details about the error or contextual data.
        '''
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        '''String representation of the error.'''
        if self.details:
            return f"{self.message} - {self.details}"
        return self.message

